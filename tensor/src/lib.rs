mod fmt;
mod host;
mod transform;

use digit_layout::DigitLayout;
use ndarray_layout::{ArrayLayout, Endian::BigEndian};
use std::{
    borrow::Cow,
    ops::{Deref, DerefMut},
};

pub extern crate digit_layout;
pub extern crate ndarray_layout;

#[derive(Clone)]
pub struct Tensor<T, const N: usize> {
    dt: DigitLayout,
    layout: ArrayLayout<N>,
    item: T,
}

impl<const N: usize> Tensor<usize, N> {
    pub fn new(dt: DigitLayout, shape: &[usize]) -> Self {
        let shape = match dt.group_size() {
            1 => Cow::Borrowed(shape),
            g => {
                let mut shape = shape.to_vec();
                let last = shape.last_mut().unwrap();
                assert_eq!(*last % g, 0);
                *last /= g;
                Cow::Owned(shape)
            }
        };

        let element_size = dt.nbytes();
        let layout = ArrayLayout::new_contiguous(&shape, BigEndian, element_size);
        let size = layout.num_elements() * element_size;
        Self {
            dt,
            layout,
            item: size,
        }
    }

    pub fn contiguous_of<U, const M: usize>(tensor: &Tensor<U, M>) -> Self {
        let dt = tensor.dt;
        let element_size = dt.nbytes();
        let layout = ArrayLayout::new_contiguous(tensor.layout.shape(), BigEndian, element_size);
        let size = layout.num_elements() * element_size;
        Self {
            dt,
            layout,
            item: size,
        }
    }
}

impl<T, const N: usize> Tensor<T, N> {
    pub const fn dt(&self) -> DigitLayout {
        self.dt
    }

    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    pub const fn layout(&self) -> &ArrayLayout<N> {
        &self.layout
    }

    pub fn take(self) -> T {
        self.item
    }

    pub const fn get(&self) -> &T {
        &self.item
    }

    pub fn get_mut(&mut self) -> &mut T {
        &mut self.item
    }

    pub fn is_contiguous(&self) -> bool {
        match self.layout.merge_be(0, self.layout.ndim()) {
            Some(layout) => {
                let &[s] = layout.strides() else {
                    unreachable!()
                };
                s == self.dt.nbytes() as isize
            }
            None => false,
        }
    }
}

impl<T, const N: usize> Tensor<T, N> {
    pub fn as_ref(&self) -> Tensor<&T, N> {
        Tensor {
            dt: self.dt,
            layout: self.layout.clone(),
            item: &self.item,
        }
    }

    pub fn as_mut(&mut self) -> Tensor<&mut T, N> {
        Tensor {
            dt: self.dt,
            layout: self.layout.clone(),
            item: &mut self.item,
        }
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Tensor<U, N> {
        let Self {
            dt,
            layout,
            item: data,
        } = self;
        Tensor {
            dt,
            layout,
            item: f(data),
        }
    }
}

impl<T: Deref, const N: usize> Tensor<T, N> {
    pub fn as_deref(&self) -> Tensor<&<T as Deref>::Target, N> {
        Tensor {
            dt: self.dt,
            layout: self.layout.clone(),
            item: self.item.deref(),
        }
    }
}

impl<T: DerefMut, const N: usize> Tensor<T, N> {
    pub fn as_deref_mut(&mut self) -> Tensor<&mut <T as Deref>::Target, N> {
        Tensor {
            dt: self.dt,
            layout: self.layout.clone(),
            item: self.item.deref_mut(),
        }
    }
}
