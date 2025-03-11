use crate::{Context, ObjId, Tensor, VirtualMachine};

pub trait Rearrange: VirtualMachine {
    fn rearrange(&self, stack: ObjId, y: &mut Tensor<Self>, x: &Tensor<Self>);
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: Rearrange + ?Sized,
{
    pub fn rearrange(&self, y: &mut Tensor<VM>, x: &Tensor<VM>) {
        self.vm.rearrange(self.stack(), y, x);
    }
}

#[cfg(test)]
impl Rearrange for crate::test::TestVM {
    fn rearrange(&self, stack: ObjId, y: &mut Tensor<Self>, x: &Tensor<Self>) {
        assert_eq!(y.dt(), x.dt());
        assert_eq!(y.shape(), x.shape());

        self.launch(
            stack,
            format!("rearrange(mut %{}, %{})", y.blob().id(), x.blob().id(),),
        )
    }
}
