use super::{NNTensor, NNTensorId};

pub type BackwardTensorOf<VM> = BackwardTensor<<VM as crate::VirtualMachine>::Memory>;

pub enum BackwardTensor<T> {
    Saved(NNTensor<T>),
    Droped(NNTensorId),
    Gradient(NNTensorId),
}

impl<T> NNTensor<T> {
    pub fn save(&self) -> BackwardTensor<T> {
        BackwardTensor::Saved(self.clone())
    }

    pub fn drop(&self) -> BackwardTensor<T> {
        BackwardTensor::Droped(self.id())
    }

    pub fn grad(&self) -> BackwardTensor<T> {
        BackwardTensor::Gradient(self.id())
    }
}
