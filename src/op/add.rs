use crate::{Context, ObjId, Tensor, VirtualMachine};

pub trait Add: VirtualMachine {
    fn add(&self, stack: ObjId, y: &mut Tensor<Self>, x: &Tensor<Self>);
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: Add + ?Sized,
{
    pub fn add(&self, y: &mut Tensor<VM>, x: &Tensor<VM>) {
        self.vm().add(self.stack(), y, x)
    }
}

#[cfg(test)]
impl Add for crate::test::TestVM {
    fn add(&self, stack: ObjId, y: &mut Tensor<Self>, x: &Tensor<Self>) {
        assert_eq!(y.dt(), x.dt());
        assert_eq!(y.shape(), x.shape());

        self.launch(
            stack,
            format!("add(mut %{}, %{})", y.blob().id(), x.blob().id(),),
        )
    }
}
