use downcast_rs::{Downcast, impl_downcast};
use std::{fmt, ops::BitOr};

pub mod add;
pub mod gelu;
pub mod gemm;
pub mod layer_norm;
pub mod rms_norm;
pub mod swiglu;

pub trait Operator {
    type Tensor;

    fn name(&self) -> String;
    fn args(&self) -> &[Access];
    fn launch(&self, tensors: &[&Self::Tensor], args: Box<dyn Args>);
}

pub trait Args: Downcast + fmt::Display {}
impl_downcast!(Args);
impl<T: Copy + fmt::Display + 'static> Args for T {}

#[derive(Clone, Copy)]
pub struct Empty;
impl fmt::Display for Empty {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{}}")
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Access {
    R,
    W,
    RW,
}

impl Access {
    pub fn may_read(self) -> bool {
        matches!(self, Self::R | Self::RW)
    }

    pub fn may_write(self) -> bool {
        matches!(self, Self::W | Self::RW)
    }
}

impl BitOr for Access {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        let r = self.may_read() || rhs.may_read();
        let w = self.may_write() || rhs.may_write();
        match (r, w) {
            (true, false) => Access::R,
            (false, true) => Access::W,
            (true, true) => Access::RW,
            (false, false) => unreachable!(),
        }
    }
}
