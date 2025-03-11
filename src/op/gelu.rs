use crate::{Context, ObjId, Tensor, VirtualMachine};

pub trait GeLU: VirtualMachine {
    fn gelu(&self, stack: ObjId, up: &mut Tensor<Self>);
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: GeLU + ?Sized,
{
    pub fn gelu(&self, up: &mut Tensor<VM>) {
        self.vm.gelu(self.stack(), up)
    }
}

#[cfg(test)]
impl GeLU for crate::test::TestVM {
    fn gelu(&self, stack: ObjId, up: &mut Tensor<Self>) {
        assert_eq!(up.shape().len(), 2);

        self.launch(stack, format!("gelu(mut %{})", up.blob().id()))
    }
}
