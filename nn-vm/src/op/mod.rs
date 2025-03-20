mod access;
mod register;

use crate::VirtualMachine;
use downcast_rs::{Downcast, impl_downcast};
use std::{any::Any, fmt};

pub mod gemm;
pub mod layer_norm;
pub mod rearrange;
pub mod rms_norm;

pub use access::Access;
pub use register::OpRegister;

pub trait Args: Downcast + fmt::Display {}
impl_downcast!(Args);

pub struct Empty;
impl Args for Empty {}
impl fmt::Display for Empty {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{}}")
    }
}

impl Args for f32 {}

pub trait Operator<VM: VirtualMachine>: Any {
    fn name(&self) -> String;
    fn args(&self) -> &[Access];
    fn launch(&self, tensors: &[&VM::Tensor], args: Box<dyn Args>);
}
