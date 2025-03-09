use crate::Blob;
use digit_layout::DigitLayout;
use ndarray_layout::{ArrayLayout, Endian::BigEndian};

#[derive(Clone)]
pub struct Tensor<B: Blob> {
    pub dt: DigitLayout,
    pub layout: ArrayLayout<4>,
    pub blob: B,
}

impl Blob for () {}

impl Tensor<()> {
    pub fn new(dt: DigitLayout, shape: &[usize]) -> Self {
        let layout = ArrayLayout::new_contiguous(shape, BigEndian, dt.nbytes());
        Self {
            dt,
            layout,
            blob: (),
        }
    }
}

impl<B: Blob> Tensor<B> {
    pub fn merge(&self, start: usize, len: usize) -> Option<Self> {
        let &Self {
            dt,
            ref layout,
            blob,
        } = self;
        layout
            .merge_be(start, len)
            .map(|layout| Self { dt, layout, blob })
    }

    pub fn tile(&self, axis: usize, tiles: &[usize]) -> Self {
        let &Self {
            dt,
            ref layout,
            blob,
        } = self;
        let layout = layout.tile_be(axis, tiles);
        Self { dt, layout, blob }
    }

    pub fn transpose(&self, perm: &[usize]) -> Self {
        let &Self {
            dt,
            ref layout,
            blob,
        } = self;
        let layout = layout.transpose(perm);
        Self { dt, layout, blob }
    }

    pub fn split<'a>(&'a self, axis: usize, parts: &'a [usize]) -> impl Iterator<Item = Self> + 'a {
        self.layout.split(axis, parts).map(|layout| Self {
            layout,
            dt: self.dt,
            blob: self.blob,
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
