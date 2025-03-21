use crate::{Access, Args, Operator};
use std::fmt;

pub trait Gemm: 'static + Sized {
    type Tensor;

    fn new() -> Self;
    fn launch(&self, c: &Self::Tensor, beta: f32, a: &Self::Tensor, b: &Self::Tensor, alpha: f32);

    fn op() -> Box<dyn Operator<Tensor = Self::Tensor>> {
        Box::new(GemmOp(Self::new()))
    }
}

#[derive(Clone, Copy)]
pub struct Scale {
    pub alpha: f32,
    pub beta: f32,
}
impl fmt::Display for Scale {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let Self { alpha, beta } = self;
        writeln!(f, "α: {alpha}, β: {beta}")
    }
}

pub struct GemmOp<Op: Gemm>(Op);

pub const NAME: &str = "gemm";

impl<Op: Gemm> Operator for GemmOp<Op> {
    type Tensor = Op::Tensor;

    fn name(&self) -> String {
        NAME.to_string()
    }

    fn args(&self) -> &[Access] {
        &[Access::W, Access::R, Access::R, Access::R]
    }

    fn launch(&self, tensors: &[&Self::Tensor], args: Box<dyn Args>) {
        let [c, a, b] = tensors else { unreachable!() };
        let &Scale { alpha, beta } = args.downcast_ref::<Scale>().unwrap();
        self.0.launch(c, beta, a, b, alpha)
    }
}
