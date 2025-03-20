mod ctx;
mod nn;
mod op;

use ctx::Branch;
use digit_layout::DigitLayout;

pub use ctx::{Context, Domain};
pub use nn::*;
pub use op::*;

/// 张量虚拟机是系统的框架，存储和管理系统的全部状态。
pub trait VirtualMachine: 'static + Sized {
    /// 虚拟机的张量。
    type Tensor: Tensor;

    // 进程管理

    /// 注册一个新的进程。
    ///
    /// 之后可用进程标识符创建上下文作用域。上下文作用域全部释放时自动释放进程。
    fn register(&self) -> u64;

    // 上下文管理

    fn domain<I>(&self, pid: u64, dev: u64, ops: I) -> Domain<Self>
    where
        I: IntoIterator<Item = Box<dyn Operator<Self>>>,
    {
        self.new_domain(pid, dev);
        Domain::new(pid, dev, ops, self)
    }

    /// 创建新的上下文作用域。
    fn new_domain(&self, pid: u64, dev: u64);
    /// 释放上下文作用域。
    fn drop_domain(&self, pid: u64, dev: u64);

    // 状态跟踪

    /// 记录一次陷入发生。
    fn record_trap(&self, pos: Pc, child_name: &str, nn_name: &str);
    /// 记录一次算子调用。
    fn record_call(&self, pos: Pc, op: &str, tensors: &[&Self::Tensor], args: &dyn Args);

    // 张量管理

    /// 创建一个新的张量，如果不传入元信息，则声明一个不指向实际张量的张量符号。
    fn new_tensor(&self, meta: Option<TensorMeta>) -> Self::Tensor;
    /// 在指定位置保存数据张量。
    fn save_tensor(&self, pos: Pc, tensor: Self::Tensor);
    /// 从指定位置加载数据张量。
    fn load_tensor(&self, pos: Pc) -> Self::Tensor;
    /// 释放张量。
    fn drop_tensor(&self, tensor: Self::Tensor);
}

/// 程序计数器，记录事件发生的位置。
pub struct Pc<'a> {
    /// 进程标识符。
    pub pid: u64,
    /// 设备标识符。
    pub dev: u64,
    /// 栈位置。
    pub path: &'a [Branch],
}

/// 张量元信息。
pub struct TensorMeta<'a> {
    /// 张量数据类型。
    pub dt: DigitLayout,
    /// 张量形状。
    pub shape: &'a [usize],
}

/// 虚拟机中的张量是指向“实际”张量的符号。
pub trait Tensor: Clone {
    /// 绑定符号指向，使 self 符号指向 other 符号指向的张量。
    ///
    /// ## NOTICE
    ///
    /// 在绑定之前 self 应该不指向实际张量。
    fn assign(&self, other: Self);

    fn meta(&self) -> Option<TensorMeta>;

    fn merge(self, start: usize, len: usize) -> Option<Self>;
    fn tile(self, axis: usize, tiles: &[usize]) -> Self;
    fn broadcast(self, axis: usize, times: usize) -> Self;
    fn transpose(self, perm: &[usize]) -> Self;
    fn slice(self, axis: usize, start: usize, len: usize) -> Self;
    fn index(self, axis: usize, index: usize) -> Self;
    fn split(self, axis: usize, parts: &[usize]) -> impl Iterator<Item = Self> + '_;
}

#[macro_export]
macro_rules! shape {
    ($tensor:expr => $pat:pat) => {
        let &$pat = $tensor.meta().unwrap().shape else {
            panic!()
        };
    };
}

#[macro_export]
macro_rules! split {
    ($tensor:expr => $( $name:ident ),+; [$( $part:expr ),+] @ $axis:expr) => {
        let parts = [$($part),+];
        let mut parts = $tensor.split($axis, &parts);
        $( let $name = parts.next().unwrap(); )+
        assert!(parts.next().is_none());
        drop(parts);
    };
}

#[cfg(test)]
pub mod test;
