//! The host's executor: tokio natively, the JavaScript event loop in a
//! JavaScript host (`wasm32-unknown-unknown`), where nothing crosses threads.

use std::future::Future;
use std::pin::Pin;

use futures_util::future::{AbortHandle, abortable};
use tokio::sync::oneshot;

/// `Send` natively; anything in a JavaScript host, whose futures hold JavaScript values.
#[cfg(not(js))]
pub trait MaybeSend: Send {}
#[cfg(not(js))]
impl<T: Send + ?Sized> MaybeSend for T {}

#[cfg(js)]
pub trait MaybeSend {}
#[cfg(js)]
impl<T: ?Sized> MaybeSend for T {}

/// `Sync` natively; anything in a JavaScript host.
#[cfg(not(js))]
pub trait MaybeSync: Sync {}
#[cfg(not(js))]
impl<T: Sync + ?Sized> MaybeSync for T {}

#[cfg(js)]
pub trait MaybeSync {}
#[cfg(js)]
impl<T: ?Sized> MaybeSync for T {}

/// A boxed future the host's executor can run.
#[cfg(not(js))]
pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send + 'static>>;
#[cfg(js)]
pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + 'static>>;

/// A spawned future. Dropping it detaches the future; it does not stop it.
pub(crate) struct Spawned {
    abort: AbortHandle,
    done: oneshot::Receiver<()>,
}

impl Spawned {
    /// Stop the future at its next suspension point.
    pub(crate) fn abort(&self) {
        self.abort.abort();
    }

    /// Wait until the future returned, was aborted, or panicked.
    pub(crate) async fn finished(self) {
        // A dropped sender means the future panicked, which also ends it.
        let _ = self.done.await;
    }
}

/// Run `future` on the host's executor: the current tokio runtime, or the JavaScript event loop.
pub(crate) fn spawn(future: impl Future<Output = ()> + MaybeSend + 'static) -> Spawned {
    let (finished, done) = oneshot::channel();
    let (future, abort) = abortable(future);
    let task = async move {
        let _ = future.await;
        let _ = finished.send(());
    };
    #[cfg(not(js))]
    tokio::spawn(task);
    #[cfg(js)]
    wasm_bindgen_futures::spawn_local(task);
    Spawned { abort, done }
}
