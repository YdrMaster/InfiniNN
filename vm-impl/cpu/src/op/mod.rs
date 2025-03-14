mod add;
mod gelu;
mod layer_norm;
mod mat_mul;
mod rearrange;
mod rms_norm;
mod rope;
mod softmax;
mod swiglu;
mod token_embed;

use crate::CpuVM;
use half::f16;
use std::sync::LazyLock;
use vm::Tensor;

fn ptr(tensor: &Tensor<CpuVM>) -> *const u8 {
    unsafe { tensor.blob().as_ptr().byte_offset(tensor.offset()) }
}

trait Exp_: Copy {
    fn exp_(self) -> Self;
    fn sigmoid(self) -> Self;
}

impl Exp_ for f16 {
    fn exp_(self) -> Self {
        static TABLE: LazyLock<Box<[f16]>> = LazyLock::new(|| {
            (0..=u16::MAX)
                .map(f16::from_bits)
                .map(|x| f16::from_f32(x.to_f32().exp()))
                .collect::<Box<_>>()
        });
        TABLE[self.to_bits() as usize]
    }
    fn sigmoid(self) -> Self {
        static TABLE: LazyLock<Box<[f16]>> = LazyLock::new(|| {
            (0..=u16::MAX)
                .map(f16::from_bits)
                .map(|x| f16::from_f32(x.to_f32().sigmoid()))
                .collect::<Box<_>>()
        });
        TABLE[self.to_bits() as usize]
    }
}

impl Exp_ for f32 {
    fn exp_(self) -> Self {
        self.exp()
    }
    fn sigmoid(self) -> Self {
        (1. + (-self).exp()).recip()
    }
}

impl Exp_ for f64 {
    fn exp_(self) -> Self {
        self.exp()
    }
    fn sigmoid(self) -> Self {
        (1. + (-self).exp()).recip()
    }
}
