use crate::{Context, Tensor, VirtualMachine};

pub trait Add: VirtualMachine {
    fn add(&self, y: &mut Tensor<Self>, x: &Tensor<Self>);
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: Add + ?Sized,
{
    pub fn add(&mut self, y: &mut Tensor<VM>, x: &Tensor<VM>) {
        self.vm.add(y, x)
    }
}

#[cfg(test)]
impl Add for crate::test::TestVM {
    fn add(&self, y: &mut Tensor<Self>, x: &Tensor<Self>) {
        assert_eq!(y.dt(), x.dt());
        assert_eq!(y.shape(), x.shape());

        self.launch(format!("add(mut %{}, %{})", y.blob().id(), x.blob().id(),))
    }
}
