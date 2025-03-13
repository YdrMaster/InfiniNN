use crate::{ObjId, Tensor, VirtualMachine};

pub trait Softmax: VirtualMachine {
    fn softmax(&self, stack: ObjId, att: &mut Tensor<Self>, mask: AttnMask);
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum AttnMask {
    None,
    Causal,
}
