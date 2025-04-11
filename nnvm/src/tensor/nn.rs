use super::Tensor;
use rw_rc::{LocalMut, LocalRef, RwRc};
use std::rc::Rc;
use tensor::digit_layout::DigitLayout;

pub struct NNTensor<T> {
    /// 张量
    tensor: Tensor<RwRc<T>>,
    /// 分配一个小空间用做 id，避免引入静态变量
    id: Rc<()>,
}

#[derive(Clone)]
#[repr(transparent)]
pub struct NNTensorId(Rc<()>);

impl<T> Clone for NNTensor<T> {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
            id: self.id.clone(),
        }
    }
}

impl<T> NNTensor<T> {
    pub fn dt(&self) -> DigitLayout {
        self.tensor.dt()
    }

    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    pub fn is_mutable(&self) -> bool {
        self.tensor.is_mutable()
    }

    pub fn read(&mut self) -> Tensor<LocalRef<T>> {
        self.tensor.read()
    }

    pub fn write(&mut self) -> Tensor<LocalMut<T>> {
        self.tensor.write()
    }

    pub(crate) fn id(&self) -> NNTensorId {
        NNTensorId(self.id.clone())
    }
}
