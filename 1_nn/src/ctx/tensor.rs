use super::internal::Internal;
use crate::Dim;
use std::{cell::RefCell, rc::Weak};
use tensor::digit_layout::DigitLayout;

/// 计算图层张量
pub struct Tensor<T> {
    pub(super) idx: usize,
    pub(super) ctx: Weak<RefCell<Internal<T>>>,
}

impl<T> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            idx: self.idx,
            ctx: self.ctx.clone(),
        }
    }
}

impl<T> Tensor<T> {
    #[inline]
    pub fn dt(&self) -> DigitLayout {
        self.meta().dt
    }

    #[inline]
    pub fn shape(&self) -> Box<[Dim]> {
        self.meta().shape.clone()
    }

    fn meta(&self) -> TensorMeta {
        self.ctx.upgrade().unwrap().borrow().tensor(self.idx)
    }
}

#[derive(Clone)]
pub struct TensorMeta {
    pub dt: DigitLayout,
    pub shape: Box<[Dim]>,
}

impl TensorMeta {
    pub fn new(dt: DigitLayout, shape: impl IntoIterator<Item = Dim>) -> Self {
        let mut shape = shape.into_iter().collect::<Box<_>>();
        let group = dt.group_size();
        if group > 1 {
            if let Some(dim) = shape.last_mut() {
                *dim = std::mem::replace(dim, Dim::from(0)) / group
            }
        }
        Self { dt, shape }
    }

    #[inline]
    pub const fn dt(&self) -> DigitLayout {
        self.dt
    }

    #[inline]
    pub fn shape(&self) -> &[Dim] {
        &self.shape
    }
}
