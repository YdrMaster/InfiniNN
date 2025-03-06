use digit_layout::DigitLayout;
use ndarray_layout::{ArrayLayout, Endian::BigEndian};
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

pub mod nn;

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
}

pub trait MemManager<A, B: Backend> {
    fn malloc(&self, size: usize) -> *mut B::Byte;
    fn load_mut(&self, which: A) -> *mut B::Byte;
    fn load(&self, which: A) -> *const B::Byte;
    fn drop(&self, ptr: *const B::Byte);

    fn workspace<'t>(&'t self, tensor: &'t Tensor) -> TensorGuard<'t, A, B, Self> {
        let size = tensor.layout.num_elements() * tensor.dt.nbytes();
        let ptr = self.malloc(size);
        TensorGuard {
            st: StorageTensor {
                tensor,
                ptr: Ptr::Mut(ptr),
            },
            mamager: self,
            phantom: PhantomData,
        }
    }
    fn load_tensor_mut<'t>(&'t self, which: A, tensor: &'t Tensor) -> TensorGuard<'t, A, B, Self> {
        let ptr = self.load_mut(which);
        TensorGuard {
            st: StorageTensor {
                tensor,
                ptr: Ptr::Mut(ptr),
            },
            mamager: self,
            phantom: PhantomData,
        }
    }
    fn load_tensor<'t>(&'t self, which: A, tensor: &'t Tensor) -> TensorGuard<'t, A, B, Self> {
        let ptr = self.load(which);
        TensorGuard {
            st: StorageTensor {
                tensor,
                ptr: Ptr::Const(ptr),
            },
            mamager: self,
            phantom: PhantomData,
        }
    }
}

pub struct StorageTensor<'a, B: Backend> {
    tensor: &'a Tensor,
    ptr: Ptr<B::Byte>,
}

pub struct TensorGuard<'a, A, B, M>
where
    B: Backend,
    M: MemManager<A, B> + ?Sized,
{
    st: StorageTensor<'a, B>,
    mamager: &'a M,
    phantom: PhantomData<(A, B)>,
}

impl<'a, A, B, M> Deref for TensorGuard<'a, A, B, M>
where
    B: Backend,
    M: MemManager<A, B> + ?Sized,
{
    type Target = StorageTensor<'a, B>;

    fn deref(&self) -> &Self::Target {
        &self.st
    }
}

impl<'a, A, B, M> DerefMut for TensorGuard<'a, A, B, M>
where
    B: Backend,
    M: MemManager<A, B> + ?Sized,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        assert!(matches!(self.st.ptr, Ptr::Mut(_)));
        &mut self.st
    }
}

impl<A, B, M> Drop for TensorGuard<'_, A, B, M>
where
    B: Backend,
    M: MemManager<A, B> + ?Sized,
{
    fn drop(&mut self) {
        let ptr = match self.st.ptr {
            Ptr::Mut(ptr) => ptr.cast_const(),
            Ptr::Const(ptr) => ptr,
        };
        self.mamager.drop(ptr)
    }
}

pub enum Ptr<T> {
    Mut(*mut T),
    Const(*const T),
}

impl<T> Clone for Ptr<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Mut(arg0) => Self::Mut(arg0.clone()),
            Self::Const(arg0) => Self::Const(arg0.clone()),
        }
    }
}

impl<T> Copy for Ptr<T> {}

#[derive(Clone)]
pub struct Tensor {
    dt: DigitLayout,
    layout: ArrayLayout<4>,
}

impl Tensor {
    fn new(dt: DigitLayout, shape: &[usize]) -> Self {
        Self {
            dt,
            layout: ArrayLayout::new_contiguous(shape, BigEndian, dt.nbytes()),
        }
    }

    fn broadcast(&self, axis: usize, times: usize) -> Self {
        Self {
            dt: self.dt,
            layout: self.layout.broadcast(axis, times),
        }
    }

    fn split<'a>(&'a self, axis: usize, parts: &'a [usize]) -> impl Iterator<Item = Self> + 'a {
        self.layout.split(axis, parts).map(|layout| Self {
            dt: self.dt,
            layout,
        })
    }

    fn is_contiguous(&self) -> bool {
        self.layout.merge_be(0, self.layout.ndim()).is_some()
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
