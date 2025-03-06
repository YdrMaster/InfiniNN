use crate::{Backend, Ptr};
use digit_layout::DigitLayout;
use ndarray_layout::{ArrayLayout, Endian::BigEndian};

pub struct StorageTensor<'a, B: Backend> {
    pub tensor: &'a Tensor,
    pub ptr: Ptr<B>,
}

#[derive(Clone)]
pub struct Tensor {
    pub dt: DigitLayout,
    pub layout: ArrayLayout<4>,
}

impl Tensor {
    pub fn new(dt: DigitLayout, shape: &[usize]) -> Self {
        Self {
            dt,
            layout: ArrayLayout::new_contiguous(shape, BigEndian, dt.nbytes()),
        }
    }

    pub fn broadcast(&self, axis: usize, times: usize) -> Self {
        Self {
            dt: self.dt,
            layout: self.layout.broadcast(axis, times),
        }
    }

    pub fn split<'a>(&'a self, axis: usize, parts: &'a [usize]) -> impl Iterator<Item = Self> + 'a {
        self.layout.split(axis, parts).map(|layout| Self {
            dt: self.dt,
            layout,
        })
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
