use crate::{Access, Args, Empty, Operator};

pub trait GeLU: 'static + Sized {
    type Tensor;

    fn new() -> Self;
    fn launch(&self, y: &Self::Tensor, x: &Self::Tensor);

    fn op() -> Box<dyn Operator<Tensor = Self::Tensor>> {
        Box::new(GeluOp(Self::new()))
    }
}

pub struct GeluOp<Op: GeLU>(Op);

pub const NAME: &str = "gelu";

impl<Op: GeLU> Operator for GeluOp<Op> {
    type Tensor = Op::Tensor;

    fn name(&self) -> String {
        NAME.to_string()
    }

    fn args(&self) -> &[Access] {
        &[Access::W, Access::R]
    }

    fn launch(&self, tensors: &[&Self::Tensor], args: Box<dyn Args>) {
        let [y, x] = tensors else { unreachable!() };
        args.downcast_ref::<Empty>().unwrap();
        self.0.launch(y, x)
    }
}
