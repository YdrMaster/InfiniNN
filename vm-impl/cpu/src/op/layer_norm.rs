use vm::{ObjId, Tensor, op::LayerNorm};

impl LayerNorm for crate::CpuVM {
    fn layer_norm(
        &self,
        _stack: ObjId,
        y: &mut Tensor<Self>,
        x: &Tensor<Self>,
        w: &Tensor<Self>,
        b: &Tensor<Self>,
    ) {
        assert_eq!(y.dt(), x.dt());
        assert_eq!(y.shape(), x.shape());
        assert_eq!(w.dt(), b.dt());

        todo!()
    }
}
