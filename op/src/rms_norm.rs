use crate::{Access, Args, Operator};

pub trait RmsNorm: 'static + Sized {
    type Tensor;

    fn new() -> Self;
    fn launch(&self, y: &Self::Tensor, x: &Self::Tensor, scale: &Self::Tensor, epsilon: f32);

    fn op() -> Box<dyn Operator<Tensor = Self::Tensor>> {
        Box::new(RmsNormOp(Self::new()))
    }
}

pub struct RmsNormOp<Op: RmsNorm>(Op);

pub const NAME: &str = "rms-norm";

impl<Op: RmsNorm> Operator for RmsNormOp<Op> {
    type Tensor = Op::Tensor;

    fn name(&self) -> String {
        NAME.to_string()
    }

    fn args(&self) -> &[Access] {
        &[Access::W, Access::R, Access::R]
    }

    fn launch(&self, tensors: &[&Self::Tensor], args: Box<dyn Args>) {
        let [y, x, scale] = tensors else {
            unreachable!()
        };
        let epsilon = args.downcast_ref::<f32>().unwrap();
        self.0.launch(y, x, scale, *epsilon)
    }
}
