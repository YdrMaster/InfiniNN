use crate::{LayoutManage, MemManage, Ptr, StorageTensor, Tensor, TrapTrace};
use digit_layout::DigitLayout;
use std::ops::{Deref, DerefMut};

pub(crate) trait TrapTraceExt: TrapTrace {
    fn trap(&self, ctx: impl Copy) -> TrapGuard<Self> {
        self.step_in(ctx);
        TrapGuard(self)
    }
}

impl<T: TrapTrace> TrapTraceExt for T {}

pub(crate) struct TrapGuard<'a, T: TrapTrace + ?Sized>(&'a T);

impl<T: TrapTrace + ?Sized> Drop for TrapGuard<'_, T> {
    fn drop(&mut self) {
        self.0.step_out()
    }
}

pub(crate) trait LayoutManageExt: LayoutManage {
    fn tensor(&self, which: impl Copy, dt: DigitLayout, shape: &[usize]) -> Tensor {
        let layout = self.get(which);
        assert_eq!(shape, layout.shape());
        Tensor { dt, layout }
    }

    fn set_tensor(&self, which: impl Copy, tensor: &Tensor) {
        self.set(which, tensor.layout.clone())
    }
}

impl<T> LayoutManageExt for T where T: LayoutManage {}

pub(crate) trait MemManageExt: MemManage {
    fn trap_with<T: Copy>(
        &self,
        ctx: impl Copy,
        args: &[(T, Ptr<Self::B>)],
    ) -> LaunchGuard<T, Self> {
        self.step_in(ctx);
        LaunchGuard {
            arg: args
                .iter()
                .map(|&(which, ptr)| {
                    self.push_arg(which, ptr);
                    which
                })
                .collect(),
            mamager: self,
        }
    }

    fn workspace<'t>(&'t self, tensor: &'t Tensor) -> TensorGuard<'t, Self> {
        let size = tensor.layout.num_elements() * tensor.dt.nbytes();
        let ptr = self.malloc(size);
        TensorGuard {
            st: StorageTensor { tensor, ptr },
            mamager: self,
        }
    }

    fn tensor<'t>(
        &'t self,
        which: impl Copy,
        tensor: &'t Tensor,
        mutable: bool,
    ) -> TensorGuard<'t, Self> {
        let ptr = self.load(which, mutable);
        TensorGuard {
            st: StorageTensor { tensor, ptr },
            mamager: self,
        }
    }
}

impl<T: MemManage> MemManageExt for T {}

pub(crate) struct LaunchGuard<'a, T, M>
where
    T: Copy,
    M: MemManage + ?Sized,
{
    arg: Vec<T>,
    mamager: &'a M,
}

impl<'a, T, M> Drop for LaunchGuard<'a, T, M>
where
    T: Copy,
    M: MemManage + ?Sized,
{
    fn drop(&mut self) {
        for arg in std::mem::take(&mut self.arg) {
            self.mamager.pop_arg(arg)
        }
        self.mamager.step_out()
    }
}

pub(crate) struct TensorGuard<'a, M>
where
    M: MemManage + ?Sized,
{
    st: StorageTensor<'a, M::B>,
    mamager: &'a M,
}

impl<'a, M> Deref for TensorGuard<'a, M>
where
    M: MemManage + ?Sized,
{
    type Target = StorageTensor<'a, M::B>;

    fn deref(&self) -> &Self::Target {
        &self.st
    }
}

impl<'a, M> DerefMut for TensorGuard<'a, M>
where
    M: MemManage + ?Sized,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.st
    }
}

impl<'a, M> Drop for TensorGuard<'a, M>
where
    M: MemManage + ?Sized,
{
    fn drop(&mut self) {
        self.mamager.drop(self.st.ptr)
    }
}
