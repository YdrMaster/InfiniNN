use super::Tensor;
use rw_rc::RwRc;

#[repr(transparent)]
pub struct SavedTensor<T>(Tensor<RwRc<T>>);

impl<T> From<Tensor<RwRc<T>>> for SavedTensor<T> {
    fn from(tensor: Tensor<RwRc<T>>) -> Self {
        assert!(tensor.lock());
        SavedTensor(tensor)
    }
}

impl<T> SavedTensor<T> {
    fn take(self) -> Tensor<RwRc<T>> {
        self.0
    }
}
