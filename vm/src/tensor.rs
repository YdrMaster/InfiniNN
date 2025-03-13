use crate::{Blob, VirtualMachine};
use digit_layout::DigitLayout;
use ndarray_layout::{ArrayLayout, Endian::BigEndian};
use std::sync::Arc;

pub struct Tensor<'vm, VM: VirtualMachine + ?Sized> {
    dt: DigitLayout,
    layout: ArrayLayout<4>,
    blob: Arc<BlobGuard<'vm, VM>>,
}

struct BlobGuard<'vm, VM: VirtualMachine + ?Sized> {
    vm: &'vm VM,
    blob: Option<VM::Blob>,
}

impl<'vm, VM: VirtualMachine + ?Sized> Tensor<'vm, VM> {
    pub fn new(dt: DigitLayout, shape: &[usize], blob: VM::Blob, vm: &'vm VM) -> Self {
        let layout = ArrayLayout::new_contiguous(shape, BigEndian, dt.nbytes());
        let size = layout.num_elements() * dt.nbytes();
        assert_eq!(size, blob.n_bytes());
        Self {
            dt,
            layout,
            blob: Arc::new(BlobGuard {
                vm,
                blob: Some(blob),
            }),
        }
    }
}

impl<VM: VirtualMachine + ?Sized> Clone for Tensor<'_, VM> {
    fn clone(&self) -> Self {
        Self {
            dt: self.dt,
            layout: self.layout.clone(),
            blob: self.blob.clone(),
        }
    }
}

impl<VM: VirtualMachine + ?Sized> Drop for BlobGuard<'_, VM> {
    fn drop(&mut self) {
        self.vm.free(self.blob.take().unwrap())
    }
}

impl<VM: VirtualMachine + ?Sized> Tensor<'_, VM> {
    pub const fn dt(&self) -> DigitLayout {
        self.dt
    }

    pub fn layout(&self) -> &ArrayLayout<4> {
        &self.layout
    }

    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    pub fn offset(&self) -> isize {
        self.layout.offset()
    }

    pub fn blob(&self) -> &VM::Blob {
        self.blob.blob.as_ref().unwrap()
    }

    pub fn check_dt_same(mut tensors: &[&Tensor<VM>]) -> Option<DigitLayout> {
        let mut ans = None;
        while let [head, tail @ ..] = tensors {
            tensors = tail;
            if let Some(dt) = ans {
                if head.dt != dt {
                    return None;
                }
            } else {
                ans = Some(head.dt)
            }
        }
        ans
    }
}

impl<VM: VirtualMachine + ?Sized> Tensor<'_, VM> {
    pub fn merge(self, start: usize, len: usize) -> Option<Self> {
        self.layout
            .merge_be(start, len)
            .map(|layout| self.map_layout(|_| layout))
    }

    pub fn tile(self, axis: usize, tiles: &[usize]) -> Self {
        self.map_layout(|l| l.tile_be(axis, tiles))
    }

    pub fn broadcast(self, axis: usize, times: usize) -> Self {
        self.map_layout(|l| l.broadcast(axis, times))
    }

    pub fn transpose(self, perm: &[usize]) -> Self {
        self.map_layout(|l| l.transpose(perm))
    }

    pub fn slice(self, axis: usize, start: usize, len: usize) -> Self {
        self.map_layout(|l| l.slice(axis, start, 1, len))
    }

    pub fn index(self, axis: usize, index: usize) -> Self {
        self.map_layout(|l| l.index(axis, index))
    }

    pub fn split<'a>(&'a self, axis: usize, parts: &'a [usize]) -> impl Iterator<Item = Self> + 'a {
        self.layout
            .split(axis, parts)
            .map(|layout| self.clone().map_layout(|_| layout))
    }

    fn map_layout(mut self, f: impl FnOnce(&ArrayLayout<4>) -> ArrayLayout<4>) -> Self {
        self.layout = f(&self.layout);
        self
    }
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
