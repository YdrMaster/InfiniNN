use crate::Tensor;
use digit_layout::types;
use std::fmt;

impl<const N: usize> fmt::Display for Tensor<&[u8], N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let layout = self.layout();
        let ptr = self.get().as_ptr();

        macro_rules! write_array {
            ($( $dt:path => $ty:ty )+) => {
                match self.dt() {
                    $( $dt => layout.write_array(f, ptr.cast::<DataFmt<$ty>>()), )+
                    dt => todo!("Unsupported data type {dt:?}"),
                }
            };
        }

        unsafe {
            write_array! {
                types::F32 => f32
                types::F64 => f64
                types::U32 => u32
                types::U64 => u64
            }
        }
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
struct DataFmt<T: Copy>(T);

impl fmt::Display for DataFmt<f32> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0. {
            write!(f, " ________")
        } else {
            write!(f, "{:>9.3e}", self.0)
        }
    }
}

impl fmt::Display for DataFmt<f64> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0. {
            write!(f, " ________")
        } else {
            write!(f, "{:>9.3e}", self.0)
        }
    }
}

impl fmt::Display for DataFmt<u32> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0 {
            write!(f, " ________")
        } else {
            write!(f, "{:>6}", self.0)
        }
    }
}

impl fmt::Display for DataFmt<u64> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0 {
            write!(f, " ________")
        } else {
            write!(f, "{:>6}", self.0)
        }
    }
}
