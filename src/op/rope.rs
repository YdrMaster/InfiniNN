use crate::{Context, ObjId, Tensor, VirtualMachine};

pub trait RoPE: VirtualMachine {
    fn rope(
        &self,
        stack: ObjId,
        x: &mut Tensor<Self>,
        pos: &Tensor<Self>,
        sin: &Tensor<Self>,
        cos: &Tensor<Self>,
    );
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: RoPE + ?Sized,
{
    pub fn rope(&self, x: &mut Tensor<VM>, pos: &Tensor<VM>, sin: &Tensor<VM>, cos: &Tensor<VM>) {
        self.vm.rope(self.stack(), x, pos, sin, cos)
    }
}

#[cfg(test)]
impl RoPE for crate::test::TestVM {
    fn rope(
        &self,
        stack: ObjId,
        x: &mut Tensor<Self>,
        pos: &Tensor<Self>,
        sin: &Tensor<Self>,
        cos: &Tensor<Self>,
    ) {
        let &[_, seq, dh] = x.shape() else { panic!() };
        let &[seq_] = pos.shape() else { panic!() };
        let &[_, dh_sin] = sin.shape() else { panic!() };
        let &[_, dh_cos] = sin.shape() else { panic!() };
        assert_eq!(seq, seq_);
        assert_eq!(dh, dh_sin * 2);
        assert_eq!(dh, dh_cos * 2);

        self.launch(
            stack,
            format!(
                "rope(mut %{}, %{}, %{}, %{})",
                x.blob().id(),
                pos.blob().id(),
                sin.blob().id(),
                cos.blob().id(),
            ),
        )
    }
}
