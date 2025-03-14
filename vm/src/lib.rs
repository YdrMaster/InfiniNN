mod stack;
mod tensor;

use std::{borrow::Cow, ops::Deref};

pub mod op;

pub use stack::{ObjId, StackTracer};
pub use tensor::Tensor;

#[allow(non_camel_case_types)]
pub type pid = u64;

#[allow(non_camel_case_types)]
pub type device_id = u64;

/// 人工智能虚拟系统。
pub trait VirtualMachine {
    /// 存储标识符。
    type Blob: Blob;

    /// 通信组标识符。
    type CommGroup: CommGroup;

    /// 注册 `arch` 架构的模型，创建一个进程，返回一个 `pid` 标识符。
    fn register(&self, arch: &str) -> pid;

    /// 注销 `pid` 标识符对应的进程。
    fn unregister(&self, pid: pid);

    /// 映射主机存储空间 `mem` 到系统中。
    fn map_host(&self, obj: ObjId, mem: Box<dyn Deref<Target = [u8]>>) -> Self::Blob;

    /// 获取映射得到的参数。
    fn get_mapped(&self, obj: ObjId) -> Self::Blob;

    /// 为 `obj` 对应的对象分配容量为 `size` 字节的设备存储空间，返回对象标识符。
    fn alloc(&self, obj: ObjId, size: usize) -> Self::Blob;

    /// 释放 `blob`。
    fn free(&self, blob: Self::Blob);

    /// 创建一个通信组，包含 `devices` 列表中的设备。
    fn comm(&self, devices: &[usize]) -> Self::CommGroup;
}

pub trait Value: Copy + Eq + Send + Sync + 'static {}
impl<T> Value for T where T: Copy + Eq + Send + Sync + 'static {}

pub trait Id: Value {
    fn name(&self) -> Cow<str>;
    fn idx(&self) -> Option<usize> {
        None
    }
}

impl Id for () {
    fn name(&self) -> Cow<str> {
        Cow::Borrowed("*")
    }
}

pub trait Blob {
    fn eq(l: &Self, r: &Self) -> bool;
    fn n_bytes(&self) -> usize;
}

pub trait CommGroup: Id {
    fn n_members(&self) -> usize;
}
