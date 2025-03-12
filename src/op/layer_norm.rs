use crate::{Context, ObjId, Tensor, VirtualMachine};

pub trait LayerNorm: VirtualMachine {
    fn layer_norm(
        &self,
        stack: ObjId,
        y: &mut Tensor<Self>,
        x: &Tensor<Self>,
        w: &Tensor<Self>,
        b: &Tensor<Self>,
    );
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: LayerNorm + ?Sized,
{
    pub fn layer_norm(&self, y: &mut Tensor<VM>, x: &Tensor<VM>, w: &Tensor<VM>, b: &Tensor<VM>) {
        self.vm().layer_norm(self.stack(), y, x, w, b)
    }
}

#[cfg(test)]
impl LayerNorm for crate::test::TestVM {
    fn layer_norm(
        &self,
        stack: ObjId,
        y: &mut Tensor<Self>,
        x: &Tensor<Self>,
        w: &Tensor<Self>,
        b: &Tensor<Self>,
    ) {
        assert_eq!(y.dt(), x.dt());
        assert_eq!(y.shape(), x.shape());
        assert_eq!(w.dt(), b.dt());

        self.launch(
            stack,
            format!(
                "layer_norm(mut %{}, %{}, %{}, %{})",
                y.blob().id(),
                x.blob().id(),
                w.blob().id(),
                b.blob().id(),
            ),
        )
    }
}
