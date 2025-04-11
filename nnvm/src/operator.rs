use crate::{Context, NNTensor, Node, VirtualMachine};
use std::any::Any;

/// 算子
///
/// 不区分正反向算子，所以也不要求正反向对应。
pub trait Operator<VM: VirtualMachine> {
    type Args: Any;

    fn compute(
        &self,
        inputs: impl IntoIterator<Item = NNTensor<VM>>,
        args: &Self::Args,
        ctx: &mut Context<VM>,
    ) -> Vec<NNTensor<VM>>;
}

pub struct OpNode<VM: VirtualMachine>(
    Box<dyn Fn(Vec<NNTensor<VM>>, &dyn Any, &mut Context<VM>) -> Vec<NNTensor<VM>>>,
);

impl<VM: VirtualMachine> OpNode<VM> {
    pub fn new(op: impl Operator<VM> + 'static) -> Self {
        Self(Box::new(move |inputs, args, ctx| {
            op.compute(inputs, args.downcast_ref().unwrap(), ctx)
        }))
    }
}

impl<VM: VirtualMachine> Node<VM> for OpNode<VM> {
    fn forward(
        &self,
        args: &dyn Any,
        inputs: Vec<NNTensor<VM>>,
        ctx: &mut Context<VM>,
    ) -> Vec<NNTensor<VM>> {
        (self.0)(inputs, args, ctx)
    }
}
