//! Where the [`Store`](crate::store::Store) runs.
//!
//! Natively, SQLite runs on a dedicated thread so async callers never block
//! on it, and the session file is locked so it has one owner. In a
//! JavaScript host there are no threads: jobs run inline, and SQLite's
//! storage and the session's single owner are the host's business (its
//! virtual file system).

pub(crate) use imp::Db;

#[cfg(not(js))]
mod imp {
    use std::fs::{File, OpenOptions, TryLockError};
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use std::thread;

    use tokio::sync::oneshot;

    use crate::error::{Error, Result};
    use crate::store::Store;

    type Job = Box<dyn FnOnce(&mut Store) + Send>;

    /// The storage thread. Closing drops the job sender, so the thread drains
    /// queued jobs, closes SQLite, and exits.
    pub(crate) struct Db {
        jobs: Mutex<Option<std::sync::mpsc::Sender<Job>>>,
        thread: Mutex<Option<thread::JoinHandle<()>>>,
    }

    /// Hold the exclusive lock of a session file: `<file>.lock`, beside SQLite's own files.
    ///
    /// A session has one owner. The owner allocates IDs and sequence numbers in
    /// memory and keeps committed state warm, so a second process writing the
    /// same file would corrupt it.
    fn lock(path: &Path) -> Result<File> {
        let mut name = path.as_os_str().to_owned();
        name.push(".lock");
        let file = OpenOptions::new().create(true).truncate(false).write(true).open(PathBuf::from(name))?;
        match file.try_lock() {
            Ok(()) => Ok(file),
            Err(TryLockError::WouldBlock) => Err(Error::Locked(path.display().to_string())),
            Err(TryLockError::Error(error)) => Err(error.into()),
        }
    }

    impl Db {
        pub(crate) async fn open(path: Option<PathBuf>) -> Result<Db> {
            tokio::task::spawn_blocking(move || Db::spawn(path)).await.map_err(|_| Error::Closed)?
        }

        fn spawn(path: Option<PathBuf>) -> Result<Db> {
            let (jobs, inbox) = std::sync::mpsc::channel::<Job>();
            let (opened, ready) = std::sync::mpsc::channel();
            let thread = thread::Builder::new()
                .name("durable-store".into())
                .spawn(move || {
                    let opening = path.as_deref().map(lock).transpose().and_then(|held| Ok((held, Store::open(path.as_deref())?)));
                    // The lock is released only after SQLite has closed.
                    let (_held, mut store) = match opening {
                        Ok(opened) => opened,
                        Err(error) => {
                            let _ = opened.send(Err(error));
                            return;
                        }
                    };
                    let _ = opened.send(Ok(()));
                    for job in inbox {
                        job(&mut store);
                    }
                    drop(store);
                })
                .map_err(|error| Error::Corrupt(format!("cannot start the storage thread: {error}")))?;
            ready.recv().map_err(|_| Error::Closed)??;
            Ok(Db { jobs: Mutex::new(Some(jobs)), thread: Mutex::new(Some(thread)) })
        }

        pub(crate) async fn call<R: Send + 'static>(&self, job: impl FnOnce(&mut Store) -> Result<R> + Send + 'static) -> Result<R> {
            let (reply, result) = oneshot::channel();
            let job: Job = Box::new(move |store| {
                let _ = reply.send(job(store));
            });
            self.send(job)?;
            result.await.map_err(|_| Error::Closed)?
        }

        fn send(&self, job: Job) -> Result<()> {
            let jobs = self.jobs.lock().expect("job sender lock poisoned");
            jobs.as_ref().ok_or(Error::Closed)?.send(job).map_err(|_| Error::Closed)
        }

        /// Stop accepting jobs and wait until the storage thread has closed SQLite.
        pub(crate) async fn close(&self) {
            self.jobs.lock().expect("job sender lock poisoned").take();
            let thread = self.thread.lock().expect("thread lock poisoned").take();
            if let Some(thread) = thread {
                // A panicked storage thread has nothing left to close.
                let _ = tokio::task::spawn_blocking(move || thread.join()).await;
            }
        }
    }

    impl Drop for Db {
        /// A session dropped without closing still closes SQLite and releases its file before it is gone.
        fn drop(&mut self) {
            self.jobs.get_mut().expect("job sender lock poisoned").take();
            if let Some(thread) = self.thread.get_mut().expect("thread lock poisoned").take() {
                let _ = thread.join();
            }
        }
    }
}

#[cfg(js)]
mod imp {
    use std::path::PathBuf;
    use std::sync::Mutex;

    use crate::error::{Error, Result};
    use crate::store::Store;

    /// The store, called inline. Closing drops it, which closes SQLite.
    pub(crate) struct Db {
        store: Mutex<Option<Store>>,
    }

    impl Db {
        pub(crate) async fn open(path: Option<PathBuf>) -> Result<Db> {
            Ok(Db { store: Mutex::new(Some(Store::open(path.as_deref())?)) })
        }

        pub(crate) async fn call<R: 'static>(&self, job: impl FnOnce(&mut Store) -> Result<R> + 'static) -> Result<R> {
            self.now(job)
        }

        /// Run a job right away: storage is inline, so nothing here waits.
        pub(crate) fn now<R>(&self, job: impl FnOnce(&mut Store) -> Result<R>) -> Result<R> {
            let mut store = self.store.lock().expect("store lock poisoned");
            job(store.as_mut().ok_or(Error::Closed)?)
        }

        pub(crate) async fn close(&self) {
            self.store.lock().expect("store lock poisoned").take();
        }
    }
}
