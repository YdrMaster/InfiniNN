use digit_layout::DigitLayout;
use std::mem::ManuallyDrop;

/// 张量虚拟机是系统的框架，存储和管理系统的全部状态。
pub trait VirtualMachine: 'static + Sized {
    /// 虚拟机的张量。
    type Tensor: VMTensor;

    // 进程管理

    /// 注册一个新的进程。
    ///
    /// 之后可用进程标识符创建上下文作用域。上下文作用域全部释放时自动释放进程。
    fn register(&self) -> u64;

    // 上下文管理

    /// 创建新的上下文作用域。
    fn new_domain(&self, pid: u64, dev: u64);
    /// 释放上下文作用域。
    fn drop_domain(&self, pid: u64, dev: u64);

    // 状态跟踪

    /// 记录一次陷入发生。
    // fn record_trap(&self, pc: Pc, child_name: &str, nn_name: &str);
    /// 记录一次算子调用。
    // fn record_call(&self, pc: Pc, op: &str, tensors: &[&Self::Tensor], args: &dyn Args);

    // 张量管理

    /// 创建一个新的张量，如果不传入元信息，则声明一个不指向实际张量的张量符号。
    fn new_tensor(&self, meta: Option<TensorMeta>) -> Self::Tensor;
    /// 在指定位置保存数据张量。
    fn save_tensor(&self, pc: Pc, tensor: Self::Tensor);
    /// 从指定位置加载数据张量。
    fn load_tensor(&self, pc: Pc) -> Self::Tensor;
    /// 克隆张量。
    fn clone_tensor(&self, tensor: &Self::Tensor) -> Self::Tensor;
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

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Branch {
    pub child_id: usize,
    pub loop_idx: usize,
}

/// 张量元信息。
pub struct TensorMeta<'a> {
    /// 张量数据类型。
    pub dt: DigitLayout,
    /// 张量形状。
    pub shape: &'a [usize],
}

/// 虚拟机中的张量是指向“实际”张量的符号。
pub trait VMTensor {
    fn meta(&self) -> Option<TensorMeta>;
}

pub struct Tensor<'ctx, VM: VirtualMachine> {
    internal: ManuallyDrop<VM::Tensor>,
    vm: &'ctx VM,
}

impl<VM: VirtualMachine> Clone for Tensor<'_, VM> {
    fn clone(&self) -> Self {
        // 引用计数 +1
        let Self { internal, vm } = self;
        let internal = ManuallyDrop::new(vm.clone_tensor(internal));
        Self { internal, vm }
    }
}

impl<VM: VirtualMachine> Drop for Tensor<'_, VM> {
    fn drop(&mut self) {
        // 引用计数 -1
        let Self { internal, vm } = self;
        vm.drop_tensor(unsafe { ManuallyDrop::take(internal) })
    }
}
