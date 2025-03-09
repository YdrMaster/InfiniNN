use crate::{Context, VirtualMachine, tensor::Tensor};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum AttnMask {
    None,
    Causal,
}

pub trait Softmax: VirtualMachine {
    fn softmax(&self, att: &mut Tensor<Self::Blob>, mask: AttnMask);
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: Softmax + ?Sized,
{
    pub fn softmax(&self, att: &mut Tensor<VM::Blob>, mask: AttnMask) {
        self.vm.softmax(att, mask)
    }
}
