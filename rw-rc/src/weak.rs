use crate::{Internal, RwRc, RwState};
use std::{
    cell::Cell,
    cmp,
    hash::Hash,
    rc::{Rc, Weak},
};

#[repr(transparent)]
pub struct RwWeak<T>(Weak<Internal<T>>);

impl<T> RwRc<T> {
    pub fn weak(&self) -> RwWeak<T> {
        RwWeak(Rc::downgrade(&self.rc))
    }
}

impl<T> Clone for RwWeak<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> PartialEq for RwWeak<T> {
    fn eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.0, &other.0)
    }
}

impl<T> Eq for RwWeak<T> {}

impl<T> Hash for RwWeak<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state);
    }
}

impl<T> PartialOrd for RwWeak<T> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for RwWeak<T> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        Ord::cmp(&self.0.as_ptr(), &other.0.as_ptr())
    }
}

impl<T> RwWeak<T> {
    pub fn hold(&self) -> Option<RwRc<T>> {
        self.0.upgrade().map(|rc| RwRc {
            rc,
            state: Cell::new(RwState::Hold),
        })
    }
}
