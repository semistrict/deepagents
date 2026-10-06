//! SQLite's files, kept by the host.
//!
//! One virtual file system, registered as SQLite's default, routes each
//! session's files to the `SessionFiles` the host opened it with: the
//! database, its rollback journal, and its write-ahead log. On Node that is
//! the file system; in a Durable Object, its synchronous storage. Temporary
//! files never leave memory.
//!
//! Each open session gets its own name inside SQLite, so sessions whose files
//! live in different places may share a path: Durable Objects sharing an
//! isolate, and so this instance, each keep `session.sqlite` in their own
//! storage. Two sessions of the same file conflict, as a locked file does
//! natively: same files object and path, or the same `identity` when the
//! files name one. Across processes or isolates, one owner per file is the
//! host's business.

use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Duration;

use js_sys::{Function, Object, Reflect};
use sqlite_wasm_rs::WasmOsCallback;
use sqlite_wasm_rs::utils::ffi::{
    SQLITE_CANTOPEN, SQLITE_IOERR, SQLITE_IOERR_DELETE, SQLITE_OPEN_MAIN_DB, SQLITE_OPEN_MAIN_JOURNAL, SQLITE_OPEN_WAL, sqlite3_vfs,
};
use sqlite_wasm_rs::utils::{
    MemChunksFile, OsCallback, SQLiteIoMethods, SQLiteVfs, SQLiteVfsFile, VfsAppData, VfsError, VfsFile, VfsResult, VfsStore, register_vfs,
};
use wasm_bindgen::prelude::*;

use crate::errors::{describe, named, names};

#[wasm_bindgen(typescript_custom_section)]
const SESSION_FILES: &str = r#"
/**
 * Where a session's SQLite files live. Every call is synchronous: SQLite
 * runs inline on the host's event loop. `name` is the session's path, or the
 * path plus `-journal` or `-wal`.
 */
export interface SessionFiles {
  /** Whether the file exists. */
  exists(name: string): boolean;
  /** Create the file, empty. */
  create(name: string): void;
  delete(name: string): void;
  /** Read into `into` from `offset`; returns how many bytes were read. */
  read(name: string, into: Uint8Array, offset: number): number;
  write(name: string, data: Uint8Array, offset: number): void;
  truncate(name: string, size: number): void;
  /** Make every write so far durable. */
  sync(name: string): void;
  size(name: string): number;
  /**
   * A name for the file at `path` that two files objects share exactly when
   * they reach the same file, such as its absolute path on a shared file
   * system. Without one, only this files object reaches its files.
   */
  identity?(path: string): string;
  /** Release what the files hold open, once their session has closed. */
  close?(): void;
}
"#;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "SessionFiles")]
    #[derive(Clone)]
    pub type SessionFiles;

    #[wasm_bindgen(method, catch)]
    fn exists(this: &SessionFiles, name: &str) -> Result<bool, JsValue>;
    #[wasm_bindgen(method, catch)]
    fn create(this: &SessionFiles, name: &str) -> Result<(), JsValue>;
    #[wasm_bindgen(method, catch)]
    fn delete(this: &SessionFiles, name: &str) -> Result<(), JsValue>;
    #[wasm_bindgen(method, catch)]
    fn read(this: &SessionFiles, name: &str, into: &mut [u8], offset: f64) -> Result<f64, JsValue>;
    #[wasm_bindgen(method, catch)]
    fn write(this: &SessionFiles, name: &str, data: &[u8], offset: f64) -> Result<(), JsValue>;
    #[wasm_bindgen(method, catch)]
    fn truncate(this: &SessionFiles, name: &str, size: f64) -> Result<(), JsValue>;
    #[wasm_bindgen(method, catch)]
    fn sync(this: &SessionFiles, name: &str) -> Result<(), JsValue>;
    #[wasm_bindgen(method, catch)]
    fn size(this: &SessionFiles, name: &str) -> Result<f64, JsValue>;
}

impl SessionFiles {
    /// An optional method of the host's files object.
    fn optional(&self, method: &str) -> Option<Function> {
        Reflect::get(self, &JsValue::from_str(method)).ok().and_then(|found| found.dyn_into::<Function>().ok())
    }

    fn identity(&self, path: &str) -> Result<Option<String>, JsValue> {
        match self.optional("identity") {
            None => Ok(None),
            Some(identity) => Ok(identity.call1(self, &JsValue::from_str(path))?.as_string()),
        }
    }
}

/// The suffixes SQLite gives a database's companion files.
const COMPANIONS: [&str; 2] = ["-journal", "-wal"];

fn io_error(code: i32, thrown: JsValue) -> VfsError {
    VfsError::new(code, describe(&thrown))
}

/// One open file: kept by the host under its own name, or in memory.
enum File {
    Host { files: SessionFiles, name: String },
    Memory(MemChunksFile),
}

impl VfsFile for File {
    fn read(&self, buf: &mut [u8], offset: usize) -> VfsResult<bool> {
        match self {
            File::Host { files, name } => {
                let read = files.read(name, buf, offset as f64).map_err(|thrown| io_error(SQLITE_IOERR, thrown))? as usize;
                if read >= buf.len() {
                    return Ok(true);
                }
                // A short read leaves the rest zeroed, as SQLite requires.
                buf[read..].fill(0);
                Ok(false)
            }
            File::Memory(file) => file.read(buf, offset),
        }
    }

    fn write(&mut self, buf: &[u8], offset: usize) -> VfsResult<()> {
        match self {
            File::Host { files, name } => files.write(name, buf, offset as f64).map_err(|thrown| io_error(SQLITE_IOERR, thrown)),
            File::Memory(file) => file.write(buf, offset),
        }
    }

    fn truncate(&mut self, size: usize) -> VfsResult<()> {
        match self {
            File::Host { files, name } => files.truncate(name, size as f64).map_err(|thrown| io_error(SQLITE_IOERR, thrown)),
            File::Memory(file) => file.truncate(size),
        }
    }

    fn flush(&mut self) -> VfsResult<()> {
        match self {
            File::Host { files, name } => files.sync(name).map_err(|thrown| io_error(SQLITE_IOERR, thrown)),
            File::Memory(file) => file.flush(),
        }
    }

    fn size(&self) -> VfsResult<usize> {
        match self {
            File::Host { files, name } => files.size(name).map(|size| size as usize).map_err(|thrown| io_error(SQLITE_IOERR, thrown)),
            File::Memory(file) => file.size(),
        }
    }
}

/// An open session's database file, as SQLite and the host each name it.
struct Owned {
    files: SessionFiles,
    path: String,
    identity: Option<String>,
}

impl Owned {
    /// Whether opening `path` with `files` would reach this session's file.
    fn conflicts(&self, files: &SessionFiles, path: &str, identity: Option<&str>) -> bool {
        match (self.identity.as_deref(), identity) {
            (Some(mine), Some(theirs)) => mine == theirs,
            _ => self.path == path && Object::is(&self.files, files),
        }
    }
}

#[derive(Default)]
struct Files {
    /// Each open session's database, by the name SQLite knows it as.
    sessions: HashMap<String, Owned>,
    /// Every file SQLite has open, by SQLite's name.
    open: HashMap<String, File>,
    next: u64,
}

impl Files {
    /// The host files and host name of a file SQLite names, if a session owns it.
    fn owner(&self, name: &str) -> Option<(&SessionFiles, String)> {
        if let Some(owned) = self.sessions.get(name) {
            return Some((&owned.files, owned.path.clone()));
        }
        COMPANIONS.iter().find_map(|suffix| {
            let owned = self.sessions.get(name.strip_suffix(suffix)?)?;
            Some((&owned.files, format!("{}{suffix}", owned.path)))
        })
    }

    fn host(&self, name: &str) -> VfsResult<File> {
        let (files, host) = self.owner(name).ok_or_else(|| VfsError::new(SQLITE_IOERR, format!("{name} is not open")))?;
        Ok(File::Host { files: files.clone(), name: host })
    }

    fn file(&mut self, name: &str) -> VfsResult<&mut File> {
        if !self.open.contains_key(name) {
            let file = self.host(name)?;
            self.open.insert(name.to_owned(), file);
        }
        Ok(self.open.get_mut(name).expect("inserted above"))
    }
}

type AppData = RefCell<Files>;

struct Store;

impl VfsStore<File, AppData> for Store {
    fn add_file(vfs: *mut sqlite3_vfs, name: &str, flags: i32) -> VfsResult<()> {
        let mut files = unsafe { Self::app_data(vfs) }.borrow_mut();
        let durable = flags & (SQLITE_OPEN_MAIN_DB | SQLITE_OPEN_MAIN_JOURNAL | SQLITE_OPEN_WAL) != 0;
        let file = if durable {
            let (host, host_name) = files.owner(name).ok_or_else(|| VfsError::new(SQLITE_CANTOPEN, format!("no session files for {name}")))?;
            host.create(&host_name).map_err(|thrown| io_error(SQLITE_IOERR, thrown))?;
            File::Host { files: host.clone(), name: host_name }
        } else {
            File::Memory(MemChunksFile::default())
        };
        files.open.insert(name.to_owned(), file);
        Ok(())
    }

    fn contains_file(vfs: *mut sqlite3_vfs, name: &str) -> VfsResult<bool> {
        let files = unsafe { Self::app_data(vfs) }.borrow();
        if files.open.contains_key(name) {
            return Ok(true);
        }
        match files.owner(name) {
            Some((host, host_name)) => host.exists(&host_name).map_err(|thrown| io_error(SQLITE_IOERR, thrown)),
            None => Ok(false),
        }
    }

    fn delete_file(vfs: *mut sqlite3_vfs, name: &str) -> VfsResult<()> {
        let mut files = unsafe { Self::app_data(vfs) }.borrow_mut();
        if let Some(File::Memory(_)) = files.open.remove(name) {
            return Ok(());
        }
        match files.owner(name) {
            Some((host, host_name)) => host.delete(&host_name).map_err(|thrown| io_error(SQLITE_IOERR_DELETE, thrown)),
            None => Ok(()),
        }
    }

    fn with_file<F: Fn(&File) -> VfsResult<i32>>(vfs_file: &SQLiteVfsFile, f: F) -> VfsResult<i32> {
        let name = unsafe { vfs_file.name() };
        let mut files = unsafe { Self::app_data(vfs_file.vfs) }.borrow_mut();
        f(files.file(name)?)
    }

    fn with_file_mut<F: Fn(&mut File) -> VfsResult<i32>>(vfs_file: &SQLiteVfsFile, f: F) -> VfsResult<i32> {
        let name = unsafe { vfs_file.name() };
        let mut files = unsafe { Self::app_data(vfs_file.vfs) }.borrow_mut();
        f(files.file(name)?)
    }
}

struct IoMethods;

impl SQLiteIoMethods for IoMethods {
    type File = File;
    type AppData = AppData;
    type Store = Store;

    const VERSION: i32 = 1;
}

struct Vfs;

impl SQLiteVfs<IoMethods> for Vfs {
    const VERSION: i32 = 1;

    fn sleep(dur: Duration) {
        WasmOsCallback::sleep(dur);
    }

    fn random(buf: &mut [u8]) {
        WasmOsCallback::random(buf);
    }

    fn epoch_timestamp_in_ms() -> i64 {
        WasmOsCallback::epoch_timestamp_in_ms()
    }
}

thread_local! {
    /// The registered file system's state, which SQLite also reaches through the VFS.
    static FILES: &'static VfsAppData<AppData> = install();
}

fn install() -> &'static VfsAppData<AppData> {
    let vfs = register_vfs::<IoMethods, Vfs>("deepagents-durable", RefCell::default(), true).expect("registering the host file system");
    // Safety: registered just above with this app data type, and never unregistered.
    unsafe { Store::app_data(vfs) }
}

/// Route a session's files at `path` to `files`; returns the name SQLite opens it by.
pub fn claim(path: &str, files: SessionFiles) -> Result<String, JsValue> {
    let identity = files.identity(path)?;
    FILES.with(|state| {
        let mut state = state.borrow_mut();
        if state.sessions.values().any(|owned| owned.conflicts(&files, path, identity.as_deref())) {
            return Err(named(names::SESSION_LOCKED, &format!("session file is open elsewhere: {path}")));
        }
        state.next += 1;
        let name = format!("/session-{}", state.next);
        state.sessions.insert(name.clone(), Owned { files, path: path.to_owned(), identity });
        Ok(name)
    })
}

/// Forget a closed session's files, and let them release what they hold.
pub fn release(name: &str) {
    let owned = FILES.with(|state| {
        let mut state = state.borrow_mut();
        state.open.retain(|open, _| open != name && !COMPANIONS.iter().any(|suffix| open.strip_suffix(suffix) == Some(name)));
        state.sessions.remove(name)
    });
    if let Some(owned) = owned
        && let Some(close) = owned.files.optional("close")
    {
        // Closing is best effort: the session's data is already durable.
        let _ = close.call0(&owned.files);
    }
}
