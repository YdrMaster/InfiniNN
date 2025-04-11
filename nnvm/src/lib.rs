mod operator;
mod tensor;

use std::any::Any;

pub use operator::{OpNode, Operator};
pub use tensor::{NNTensor, NNTensorId, Tensor};

pub trait VirtualMachine {
    type Memory;

    fn alloc(&self, size: usize) -> Self::Memory;
    fn dealloc(&self, memory: Self::Memory);
}

pub trait NeuralNetwork<VM: VirtualMachine> {
    type Init;
    type Args: Any;
    type BackArgs: Any;

    /// 初始化
    fn init(init: Self::Init) -> Self;

    /// 尝试根据参数生成反向算子
    fn backward_op(&self, _args: &Self::Args) -> Option<Box<dyn Node<VM>>> {
        None
    }

    /// 前向传播
    fn forward(
        &self,
        args: &Self::Args,
        inputs: impl IntoIterator<Item = NNTensor<VM>>,
        ctx: Ctx<VM>,
    ) -> ForwardOutput<VM, Self>;
}

pub struct ForwardOutput<VM, NN>
where
    VM: VirtualMachine,
    NN: NeuralNetwork<VM> + ?Sized,
{
    pub tensors: Vec<NNTensor<VM>>,
    pub backward: Option<OpNode<VM>>,
    pub backward_args: Option<NN::BackArgs>,
}

pub trait Node<VM: VirtualMachine> {
    fn forward(
        &self,
        args: &dyn Any,
        inputs: Vec<NNTensor<VM>>,
        ctx: &mut Context<VM>,
    ) -> Vec<NNTensor<VM>>;
}

pub struct Context<VM: VirtualMachine> {
    /// 上下文位置
    path: String,
    /// 虚拟机的引用
    ///
    /// TODO: 也可能是 Domain 的引用，如果有这么一层
    vm: VM,
    recorder: Option<Recorder<VM>>,
}

pub struct Ctx<'ctx, VM: VirtualMachine> {
    ctx: &'ctx mut Context<VM>,
    recorder: Option<&'ctx mut Recorder<VM>>,
}

impl<VM: VirtualMachine> Ctx<'_, VM> {
    fn trap<T>(
        &mut self,
        sub: impl AsRef<str>,
        recorder: Option<&mut Recorder<VM>>,
        f: impl FnOnce(Ctx<VM>) -> T,
    ) -> T {
        let sub = sub.as_ref();

        self.ctx.path.push('.');
        self.ctx.path.push_str(sub);

        let ans = f(Ctx {
            ctx: self.ctx,
            recorder,
        });

        assert!(self.ctx.path.ends_with(sub));
        self.ctx.path.truncate(self.ctx.path.len() - sub.len() - 1);

        ans
    }
}

struct Recorder<VM: VirtualMachine> {
    forward: Vec<TopoNode<VM>>,
    backward: Option<Vec<TopoNode<VM>>>,
}

impl<VM: VirtualMachine> Recorder<VM> {
    fn new() -> Self {
        Self {
            forward: Vec::new(),
            backward: Some(Vec::new()),
        }
    }
}

struct TopoNode<VM: VirtualMachine> {
    node: Box<dyn Node<VM>>,
    args: Option<Box<dyn Any>>,
    inputs: NNTensorId,
    outputs: NNTensorId,
}
