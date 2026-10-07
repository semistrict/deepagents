// Cross-checks durable SQLite files against pi-durable itself.
//
//   PI_ROOT=~/src/pi-interop node --conditions=source --experimental-strip-types \
//     libs/durable/interop/interop.ts write <file>   # write a session with pi's Session API
//   ... interop.ts dump <file>                       # print every record and document as canonical JSON
//
// `cargo run --example interop -- dump|write <file>` does the same with the Rust core;
// `make interop` compares both dumps of files written by either side.
import { DatabaseSync } from "node:sqlite";

const root = process.env.PI_ROOT;
if (root === undefined) throw new Error("Set PI_ROOT to a pi checkout with dependencies installed");
const durable = await import(`${root}/packages/durable/src/index.ts`);
const { openNodeSqliteStorage } = await import(`${root}/packages/durable/src/storage/sqlite/node.ts`);
const { BACKGROUND_CONTEXT: context } = await import(`${root}/packages/chord/src/context/index.ts`);

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
const PAGE = 1000;

/** Stable JSON: object keys sorted, so dumps compare byte for byte. */
function canonical(value: Json): string {
	if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
	if (value !== null && typeof value === "object") {
		const keys = Object.keys(value).sort();
		return `{${keys.map((key) => `${JSON.stringify(key)}:${canonical(value[key]!)}`).join(",")}}`;
	}
	return JSON.stringify(value);
}

async function all<T>(scan: (cursor: unknown) => Promise<{ items: T[]; next?: unknown }>): Promise<T[]> {
	const items: T[] = [];
	let cursor: unknown;
	do {
		const page = await scan(cursor);
		items.push(...page.items);
		cursor = page.next;
	} while (cursor !== undefined);
	return items;
}

async function dump(file: string): Promise<void> {
	const database = new DatabaseSync(file, { readOnly: true });
	const { seq } = database.prepare("SELECT next_seq - 1 AS seq FROM durable_metadata").get() as { seq: number };
	database.close();

	const storage = await openNodeSqliteStorage(file);
	const conversations = await all((cursor) => storage.scanConversations({}, PAGE, cursor, context));
	const entries: Record<string, Json> = {};
	const contexts: Record<string, Json> = {};
	for (const conversation of conversations) {
		const visible = await all((cursor) => storage.scanEntries({ conversationId: conversation.id }, PAGE, cursor, context));
		visible.reverse();
		const withSeq = [];
		for (const entry of visible) {
			const stored = await storage.entry(entry.id, context);
			withSeq.push({ commitSeq: stored.commitSeq, record: entry });
		}
		entries[conversation.id] = withSeq;
		const marker = await storage.findLatestHeadMarker(conversation.id, undefined, context);
		const range = marker === undefined ? visible : visible.filter((entry) => entry.id >= marker.head && entry.head === undefined);
		contexts[conversation.id] = [...(marker === undefined ? [] : [marker.id]), ...range.map((entry) => entry.id)];
	}
	const tasks = await all((cursor) => storage.scanTasks({}, PAGE, cursor, context));
	const submissions = await all((cursor) => storage.scanSubmissions({}, PAGE, cursor, context));

	const scopes = [
		{ kind: "session" },
		...conversations.map((conversation: { id: number }) => ({ kind: "conversation", conversationId: conversation.id })),
		...tasks.map((task: { id: number }) => ({ kind: "task", taskId: task.id })),
	];
	const documents = [];
	const history = [];
	for (const scope of scopes) {
		const records = await all((cursor) => storage.scanDocuments({ scope, at: "current" }, PAGE, cursor, context));
		for (const record of records) {
			const stored = await storage.document(record.id, "current", context);
			documents.push({ record, value: stored.value, version: stored.version });
			if (scope.kind !== "conversation" || record.history !== "rewindable") continue;
			for (let at = record.createdAt; at <= seq; at++) {
				const past = await storage.document(record.id, at, context);
				history.push({ at, id: record.id, value: past.value });
			}
		}
	}
	documents.sort((a, b) => a.record.id - b.record.id);
	await storage.close(context);
	process.stdout.write(`${canonical({ seq, conversations, entries, context: contexts, tasks, submissions, documents, history } as Json)}\n`);
}

async function write(file: string): Promise<void> {
	const session = durable.createSession(await openNodeSqliteStorage(file));
	const Notes = durable.defineDoc({
		kind: "interop.notes",
		version: 1,
		scope: "conversation",
		history: "rewindable",
		fork: "asOf",
		initial: () => ({ text: "", tags: [] }),
	});
	const Settings = durable.defineDoc({ kind: "interop.settings", version: 2, scope: "session", initial: () => ({ theme: "dark" }) });
	const Scratch = durable.defineDoc({ kind: "interop.scratch", version: 1, scope: "task", initial: () => ({ lines: [] }) });
	const Work = durable.defineTask({
		name: "interop.work",
		version: 3,
		initial: (input: { n: number }) => ({ phase: "start", n: input.n }),
		phases: { start: async () => {} },
		abort: async () => {},
	});

	const root = await session.commit((tx: any) => tx.createRootConversation(), context);
	const first = await session.commit(async (tx: any) => {
		const user = await tx.appendEntry(root.id, {
			kind: "pi.user",
			model: [{ role: "user", content: "What is the capital of France?", timestamp: 1 }],
		});
		(await tx.doc(Notes, root.id)).text = "Par";
		(await tx.doc(Settings)).theme = "light";
		return user;
	}, context);
	await session.commit(async (tx: any) => {
		await tx.appendEntry(root.id, { kind: "pi.assistant", data: { answer: "Paris" } });
		const notes = await tx.doc(Notes, root.id);
		notes.text += "is";
		notes.tags.push("geo", "europe");
	}, context);

	const fork = await session.commit(
		(tx: any) => tx.forkConversation(root.id, first.id, { ownership: { kind: "ownerless" } }),
		context,
	);
	const task = await session.commit((tx: any) => tx.createTask(Work, { n: 7 }, { ownership: { kind: "conversation" }, conversationId: root.id }), context);
	await session.commit(async (tx: any) => {
		const record = await tx.task(task);
		await tx.createConversation({ ownership: { kind: "task", taskId: task } });
		(await tx.doc(Scratch, task)).lines.push("started");
		tx.setTask({ ...record, state: { status: "running", checkpoint: { phase: "start", n: 7 } } });
	}, context);
	await session.commit(async (tx: any) => {
		const record = await tx.task(task);
		tx.setTask({ ...record, state: { status: "terminal", outcome: { status: "completed", result: { total: 42 } } } });
		await tx.retireDoc(Scratch, task);
	}, context);

	const submission = await session.commit(
		(tx: any) => tx.createSubmission({ conversationId: root.id, requestId: "req-1", type: "input", status: "queued" }),
		context,
	);
	await session.commit(async (tx: any) => {
		const summary = await tx.appendEntry(fork.id, { kind: "pi.compaction", data: { summary: "asked about France" }, head: first.id });
		tx.settleSubmission(submission.id, { status: "unanswered", reason: "aborted" });
		(await tx.doc(Notes, root.id)).tags.splice(0, 1);
		return summary;
	}, context);
	await session.close(context);
}

const [command, file] = process.argv.slice(2);
if (file === undefined) throw new Error("usage: interop.ts dump|write <file>");
if (command === "dump") await dump(file);
else if (command === "write") await write(file);
else throw new Error(`unknown command ${command}`);
