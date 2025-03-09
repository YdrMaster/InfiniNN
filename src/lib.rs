mod context;
mod tensor;
mod test;

use std::ops::Deref;

pub mod nn;
pub mod op;

pub use context::{Context, Exec, Map, Mapping, ObjId};
pub use tensor::Tensor;

#[allow(non_camel_case_types)]
pub type pid = u64;

/// 人工智能虚拟系统。
pub trait VirtualMachine {
    /// 存储标识符。
    type Blob: Blob;

    /// 通信组标识符。
    type CommGroup: CommGroup;

    /// 获取虚拟机管理的设备数量。
    fn num_devices(&self) -> usize;

    /// 注册 `arch` 架构的模型，创建一个进程，返回一个 `pid` 标识符。
    fn register(&self, arch: &str) -> pid;

    /// 注销 `pid` 标识符对应的进程。
    fn unregister(&self, pid: pid);

    /// 映射主机存储空间 `mem` 到系统中。
    fn map_host(&self, obj: ObjId, mem: Box<dyn Deref<Target = [u8]>>) -> Self::Blob;

    /// 获取映射得到的参数。
    fn get_mapped(&self, obj: ObjId) -> Self::Blob;

    /// 为 `obj` 对应的对象分配容量为 `size` 字节的主机存储空间，返回对象标识符。
    fn alloc_host(&self, obj: ObjId, size: usize) -> Self::Blob;

    /// 为 `obj` 对应的对象分配容量为 `size` 字节的设备存储空间，返回对象标识符。
    fn alloc(&self, obj: ObjId, size: usize) -> Self::Blob;

    /// 重新借用 blob 的一个副本，即引用计数 +1。
    fn retain(&self, obj: &Self::Blob) -> Self::Blob;

    /// 释放 `blob` 的一个副本，即引用计数 -1。
    fn release(&self, blob: Self::Blob);

    /// 创建一个通信组，包含 `devices` 列表中的设备。
    fn comm(&self, devices: &[usize]) -> Self::CommGroup;
}

pub trait Id: Copy + Eq + Send + Sync + 'static {
    fn from_slice(slice: &[u8]) -> Self {
        assert_eq!(size_of::<Self>(), slice.len());
        unsafe { slice.as_ptr().cast::<Self>().read_unaligned() }
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts((&raw const *self).cast(), size_of_val(self)) }
    }
}

impl<T: Copy + Eq + Send + Sync + 'static> Id for T {}

pub trait Blob {
    fn eq(l: &Self, r: &Self) -> bool;
    fn n_bytes(&self) -> usize;
}

pub trait CommGroup: Id {
    fn n_members(&self) -> usize;
}
