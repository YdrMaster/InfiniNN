use crate::{Context, Tensor, VirtualMachine};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum AttnMask {
    None,
    Causal,
}

pub trait Softmax: VirtualMachine {
    fn softmax(&self, att: &mut Tensor<Self>, mask: AttnMask);
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: Softmax + ?Sized,
{
    pub fn softmax(&self, att: &mut Tensor<VM>, mask: AttnMask) {
        self.vm.softmax(att, mask)
    }
}
