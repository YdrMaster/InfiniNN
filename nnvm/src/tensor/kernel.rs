use rwrc::{LocalMut, LocalRef, RwRc};

type TenosrMut<'a, T> = super::Tensor<LocalMut<'a, T>>;
type TenosrRef<'a, T> = super::Tensor<LocalRef<'a, T>>;

pub type KernelTensorOf<'a, VM> = KernelTensor<'a, <VM as crate::VirtualMachine>::Memory>;

pub enum KernelTensor<'a, T> {
    Mut(TenosrMut<'a, T>),
    Ref(TenosrRef<'a, T>),
    Inplace(usize),
}

impl<T> KernelTensor<'_, T> {
    pub fn inplace(i: usize) -> Self {
        KernelTensor::Inplace(i)
    }
}

impl<T> super::Tensor<RwRc<T>> {
    pub fn kernel_mut(&mut self) -> KernelTensor<T> {
        KernelTensor::Mut(self.write())
    }

    pub fn kernel_ref(&self) -> KernelTensor<T> {
        KernelTensor::Ref(self.read())
    }
}
