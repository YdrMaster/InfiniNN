use crate::{Backend, MemManager, StorageTensor, Tensor};
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

pub(crate) trait MemManagerExt<A, B: Backend>: MemManager<A, B> {
    fn workspace<'t>(&'t self, tensor: &'t Tensor) -> TensorGuard<'t, A, B, Self> {
        let size = tensor.layout.num_elements() * tensor.dt.nbytes();
        let ptr = self.malloc(size);
        TensorGuard {
            st: StorageTensor::new_mut(tensor, ptr),
            mamager: self,
            phantom: PhantomData,
        }
    }
    fn load_tensor_mut<'t>(&'t self, which: A, tensor: &'t Tensor) -> TensorGuard<'t, A, B, Self> {
        let ptr = self.load_mut(which);
        TensorGuard {
            st: StorageTensor::new_mut(tensor, ptr),
            mamager: self,
            phantom: PhantomData,
        }
    }
    fn load_tensor<'t>(&'t self, which: A, tensor: &'t Tensor) -> TensorGuard<'t, A, B, Self> {
        let ptr = self.load(which);
        TensorGuard {
            st: StorageTensor::new_const(tensor, ptr),
            mamager: self,
            phantom: PhantomData,
        }
    }
}

impl<T, A, B> MemManagerExt<A, B> for T
where
    T: MemManager<A, B>,
    B: Backend,
{
}

pub(crate) struct TensorGuard<'a, A, B, M>
where
    B: Backend,
    M: MemManager<A, B> + ?Sized,
{
    st: StorageTensor<'a>,
    mamager: &'a M,
    phantom: PhantomData<(A, B)>,
}

impl<'a, A, B, M> Deref for TensorGuard<'a, A, B, M>
where
    B: Backend,
    M: MemManager<A, B> + ?Sized,
{
    type Target = StorageTensor<'a>;

    fn deref(&self) -> &Self::Target {
        &self.st
    }
}

impl<'a, A, B, M> DerefMut for TensorGuard<'a, A, B, M>
where
    B: Backend,
    M: MemManager<A, B> + ?Sized,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        assert!(self.st.mutable);
        &mut self.st
    }
}

impl<A, B, M> Drop for TensorGuard<'_, A, B, M>
where
    B: Backend,
    M: MemManager<A, B> + ?Sized,
{
    fn drop(&mut self) {
        self.mamager.drop(self.st.ptr as _)
    }
}
