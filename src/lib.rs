mod ext;
mod operators;
mod tensor;
mod test_recorder;

pub mod nn;

use ndarray_layout::ArrayLayout;
pub use tensor::{StorageTensor, Tensor};

pub trait Backend {
    type Byte;
}

pub trait TrapTrace {
    fn step_in<T: Copy>(&self, ctx: T);
    fn step_out(&self);
}

pub trait LayoutManage: TrapTrace {
    fn get<T: Copy>(&self, which: T) -> ArrayLayout<4>;
    fn set<T: Copy>(&self, which: T, layout: ArrayLayout<4>);
}

#[derive(PartialEq, Eq, Hash)]
pub enum Ptr<B: Backend> {
    Host(*const u8),
    HostMut(*mut u8),
    Mut(*mut B::Byte),
    Const(*const B::Byte),
}

impl<B: Backend> Ptr<B> {
    pub fn address(&self) -> usize {
        match *self {
            Self::Host(ptr) => ptr as _,
            Self::HostMut(ptr) => ptr as _,
            Self::Mut(ptr) => ptr as _,
            Self::Const(ptr) => ptr as _,
        }
    }
}

impl<B: Backend> Clone for Ptr<B> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<B: Backend> Copy for Ptr<B> {}

pub trait MemManage: TrapTrace {
    type B: Backend;

    fn push_arg<T: Copy>(&self, which: T, ptr: Ptr<Self::B>);
    fn pop_arg<T: Copy>(&self, which: T);

    fn malloc(&self, size: usize) -> Ptr<Self::B>;
    fn load<T: Copy>(&self, which: T, mutable: bool) -> Ptr<Self::B>;
    fn drop(&self, ptr: Ptr<Self::B>);
}
