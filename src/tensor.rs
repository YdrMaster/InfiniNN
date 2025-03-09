use crate::{Blob, VirtualMachine};
use digit_layout::DigitLayout;
use ndarray_layout::{ArrayLayout, Endian::BigEndian};

pub struct Tensor<'vm, VM: VirtualMachine + ?Sized> {
    dt: DigitLayout,
    layout: ArrayLayout<4>,
    blob: Option<VM::Blob>,
    vm: &'vm VM,
}

impl<'vm, VM: VirtualMachine + ?Sized> Tensor<'vm, VM> {
    pub fn new(dt: DigitLayout, shape: &[usize], blob: VM::Blob, vm: &'vm VM) -> Self {
        let layout = ArrayLayout::new_contiguous(shape, BigEndian, dt.nbytes());
        let size = layout.num_elements() * dt.nbytes();
        assert_eq!(size, blob.n_bytes());
        Self {
            dt,
            layout,
            blob: Some(blob),
            vm,
        }
    }
}

impl<VM: VirtualMachine + ?Sized> Clone for Tensor<'_, VM> {
    fn clone(&self) -> Self {
        Self {
            dt: self.dt,
            layout: self.layout.clone(),
            blob: Some(self.vm.retain(self.blob())),
            vm: self.vm,
        }
    }
}

impl<VM: VirtualMachine + ?Sized> Drop for Tensor<'_, VM> {
    fn drop(&mut self) {
        self.vm.release(self.blob.take().unwrap())
    }
}

impl<VM: VirtualMachine + ?Sized> Tensor<'_, VM> {
    pub fn dt(&self) -> DigitLayout {
        self.dt
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
        self.blob.as_ref().unwrap()
    }

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
