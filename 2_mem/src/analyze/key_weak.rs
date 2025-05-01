use std::{
    hash::Hash,
    rc::{Rc, Weak},
};

#[repr(transparent)]
pub struct KeyWeak<T>(Weak<T>);

impl<T> From<&Rc<T>> for KeyWeak<T> {
    fn from(value: &Rc<T>) -> Self {
        Self(Rc::downgrade(value))
    }
}

impl<T> Clone for KeyWeak<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> PartialEq for KeyWeak<T> {
    fn eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.0, &other.0)
    }
}

impl<T> Eq for KeyWeak<T> {}

impl<T> Hash for KeyWeak<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state)
    }
}

impl<T> PartialOrd for KeyWeak<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for KeyWeak<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.as_ptr().cmp(&other.0.as_ptr())
    }
}

impl<T> KeyWeak<T> {
    pub fn upgrade(&self) -> Option<Rc<T>> {
        self.0.upgrade()
    }
}
