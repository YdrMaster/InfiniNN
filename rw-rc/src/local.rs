use crate::{RwRc, RwState};
use std::ops::{Deref, DerefMut};

/// 类似 [`RwLock`](std::sync::RwLock) 的自动读状态对象。
pub struct LocalRef<'w, T>(&'w RwRc<T>, RwState);

/// 类似 [`RwLock`](std::sync::RwLock) 的自动写状态对象。
pub struct LocalMut<'w, T>(&'w mut RwRc<T>, RwState);

impl<T> RwRc<T> {
    pub fn try_read(&self) -> Option<LocalRef<T>> {
        let current = self.state.get();
        self.try_read_global()
            .map(|_| ())
            .map(|()| LocalRef(self, current))
    }

    pub fn try_write(&mut self) -> Option<LocalMut<T>> {
        let current = self.state.get();
        self.try_write_global()
            .map(|_| ())
            .map(|()| LocalMut(self, current))
    }

    pub fn read(&self) -> LocalRef<T> {
        self.try_read().unwrap()
    }

    pub fn write(&mut self) -> LocalMut<T> {
        self.try_write().unwrap()
    }
}

impl<T> Drop for LocalRef<'_, T> {
    fn drop(&mut self) {
        match self.1 {
            RwState::Hold => self.0.release(),
            RwState::Read | RwState::Write => {}
        }
    }
}

impl<T> Drop for LocalMut<'_, T> {
    fn drop(&mut self) {
        match self.1 {
            RwState::Hold => {
                self.0.release();
            }
            RwState::Read => {
                self.0.release();
                self.0.read_global();
            }
            RwState::Write => {}
        }
    }
}

impl<T> Deref for LocalRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.0.read_global()
    }
}

impl<T> Deref for LocalMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.0.read_global()
    }
}

impl<T> DerefMut for LocalMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.write_global()
    }
}
