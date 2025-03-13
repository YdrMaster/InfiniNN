use crate::Context;
use vm::{
    Tensor,
    op::{AttnMask, Softmax},
};

impl<VM, NN> Context<'_, VM, NN>
where
    VM: Softmax + ?Sized,
{
    pub fn softmax(&self, att: &mut Tensor<VM>, mask: AttnMask) {
        self.vm().softmax(self.stack(), att, mask)
    }
}
