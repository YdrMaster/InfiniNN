use crate::{Access, Args, Empty, Operator};

pub trait LayerNorm: 'static + Sized {
    type Tensor;

    fn new() -> Self;
    fn launch(&self, y: &Self::Tensor, x: &Self::Tensor, scale: &Self::Tensor, bias: &Self::Tensor);

    fn op() -> Box<dyn Operator<Tensor = Self::Tensor>> {
        Box::new(LayerNormOp(Self::new()))
    }
}

pub struct LayerNormOp<Op: LayerNorm>(Op);

pub const NAME: &str = "layer-norm";

impl<Op: LayerNorm> Operator for LayerNormOp<Op> {
    type Tensor = Op::Tensor;

    fn name(&self) -> String {
        NAME.to_string()
    }

    fn args(&self) -> &[Access] {
        &[Access::W, Access::R, Access::R, Access::R]
    }

    fn launch(&self, tensors: &[&Self::Tensor], args: Box<dyn Args>) {
        let [y, x, scale, bias] = tensors else {
            unreachable!()
        };
        args.downcast_ref::<Empty>().unwrap();
        self.0.launch(y, x, scale, bias)
    }
}
