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

    pub fn merge(&self, start: usize, len: usize) -> Option<Self> {
        let &Self { dt, ref layout } = self;
        layout
            .merge_be(start, len)
            .map(|layout| Self { dt, layout })
    }

    pub fn tile(&self, axis: usize, tiles: &[usize]) -> Self {
        let &Self { dt, ref layout } = self;
        let layout = layout.tile_be(axis, tiles);
        Self { dt, layout }
    }

    pub fn transpose(&self, perm: &[usize]) -> Self {
        let &Self { dt, ref layout } = self;
        let layout = layout.transpose(perm);
        Self { dt, layout }
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
