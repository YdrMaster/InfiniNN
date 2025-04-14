use super::Tensor;
use std::slice::{from_raw_parts, from_raw_parts_mut};

impl<'s, const N: usize> Tensor<&'s [u8], N> {
    pub fn scalar<U>(&self) -> &'s U {
        let &[] = self.layout.shape() else {
            panic!("not a scalar tensor")
        };

        unsafe { &*self.ptr() }
    }

    pub fn vector<U>(&self) -> &'s [U] {
        let &[n] = self.layout.shape() else {
            panic!("not a vector tensor")
        };
        assert!(self.is_contiguous());

        unsafe { from_raw_parts(self.ptr(), n) }
    }

    pub fn ptr<U>(&self) -> *const U {
        // assert_eq!(self.dt.nbytes(), size_of::<U>());
        self.item[self.layout.offset() as usize..].as_ptr().cast()
    }
}

impl<'s, const N: usize> Tensor<&'s mut [u8], N> {
    pub fn scalar_mut<U>(&mut self) -> &'s mut U {
        let &[] = self.layout.shape() else {
            panic!("not a scalar tensor")
        };

        unsafe { &mut *self.mut_ptr() }
    }

    pub fn vector_mut<U>(&mut self) -> &'s mut [U] {
        let &[n] = self.layout.shape() else {
            panic!("not a vector tensor: {:?}", self.layout.shape())
        };
        assert!(self.is_contiguous());

        unsafe { from_raw_parts_mut(self.mut_ptr(), n) }
    }

    pub fn mut_ptr<U>(&mut self) -> *mut U {
        // assert_eq!(self.dt.nbytes(), size_of::<U>());
        self.item[self.layout.offset() as usize..]
            .as_mut_ptr()
            .cast()
    }
}
