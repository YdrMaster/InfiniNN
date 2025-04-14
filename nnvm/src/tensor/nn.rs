use super::Tensor;
use rwrc::RwRc;
use std::{
    ops::{Deref, DerefMut},
    rc::Rc,
};

pub struct NNTensor<T> {
    /// 张量
    tensor: Tensor<RwRc<T>>,
    /// 分配一个小空间用做 id，避免引入静态变量
    id: Rc<()>,
}

#[derive(Clone)]
#[repr(transparent)]
pub struct NNTensorId(Rc<()>);

impl<T> From<Tensor<RwRc<T>>> for NNTensor<T> {
    fn from(tensor: Tensor<RwRc<T>>) -> Self {
        Self {
            tensor,
            id: Rc::new(()),
        }
    }
}

impl<T> Clone for NNTensor<T> {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
            id: self.id.clone(),
        }
    }
}

impl<T> NNTensor<T> {
    pub fn id(&self) -> NNTensorId {
        NNTensorId(self.id.clone())
    }
}

impl<T> Deref for NNTensor<T> {
    type Target = Tensor<RwRc<T>>;

    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}

impl<T> DerefMut for NNTensor<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tensor
    }
}
