mod ext;
mod tensor;

use digit_layout::DigitLayout;
use ext::MemManagerExt;
use ndarray_layout::ArrayLayout;

pub mod nn;

pub use tensor::{StorageTensor, Tensor};

pub trait Backend {
    type Byte;
    type Queue<'q>;
}

pub trait LayoutManager<A> {
    fn get(&self, which: A) -> (&[isize], isize);
    fn set(&mut self, which: A, layout: (&[isize], isize));

    fn tensor(&self, which: A, dt: DigitLayout, shape: &[usize]) -> Tensor {
        Tensor {
            dt,
            layout: {
                let (strides, offset) = self.get(which);
                ArrayLayout::new(shape, strides, offset)
            },
        }
    }
    fn set_tensor(&mut self, which: A, tensor: &Tensor) {
        self.set(which, (tensor.layout.strides(), tensor.layout.offset()))
    }
}

pub trait MemManager<A, B: Backend> {
    fn malloc(&self, size: usize) -> *mut B::Byte;
    fn load_mut(&self, which: A) -> *mut B::Byte;
    fn load(&self, which: A) -> *const B::Byte;
    fn drop(&self, ptr: *const B::Byte);
}
