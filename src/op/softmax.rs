use crate::{Context, ObjId, Tensor, VirtualMachine};

pub trait Softmax: VirtualMachine {
    fn softmax(&self, stack: ObjId, att: &mut Tensor<Self>, mask: AttnMask);
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: Softmax + ?Sized,
{
    pub fn softmax(&self, att: &mut Tensor<VM>, mask: AttnMask) {
        self.vm.softmax(self.stack(), att, mask)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum AttnMask {
    None,
    Causal,
}

#[cfg(test)]
impl Softmax for crate::test::TestVM {
    fn softmax(&self, stack: ObjId, att: &mut Tensor<Self>, mask: AttnMask) {
        assert_eq!(att.shape().len(), 3);

        let mask = match mask {
            AttnMask::None => "",
            AttnMask::Causal => ", causal",
        };
        self.launch(stack, format!("softmax(mut %{}{mask})", att.blob().id()))
    }
}
