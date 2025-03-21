use crate::{Access, Args, Empty, Operator};

pub trait Add: 'static + Sized {
    type Tensor;

    fn new() -> Self;
    fn launch(&self, c: &Self::Tensor, a: &Self::Tensor, b: &Self::Tensor);

    fn op() -> Box<dyn Operator<Tensor = Self::Tensor>> {
        Box::new(AddOp(Self::new()))
    }
}

pub struct AddOp<Op: Add>(Op);

pub const NAME: &str = "add";

impl<Op: Add> Operator for AddOp<Op> {
    type Tensor = Op::Tensor;

    fn name(&self) -> String {
        NAME.to_string()
    }

    fn args(&self) -> &[Access] {
        &[Access::W, Access::R, Access::R]
    }

    fn launch(&self, tensors: &[&Self::Tensor], args: Box<dyn Args>) {
        let [c, a, b] = tensors else { unreachable!() };
        args.downcast_ref::<Empty>().unwrap();
        self.0.launch(c, a, b)
    }
}
