use crate::op::ptr;
use digit_layout::types;
use half::f16;
use std::{
    fmt::Display,
    ops::{Add, Mul, Sub},
};
use vm::{ObjId, Tensor, op::RoPE};

impl RoPE for crate::CpuVM {
    fn rope(
        &self,
        _stack: ObjId,
        x: &mut Tensor<Self>,
        pos: &Tensor<Self>,
        sin: &Tensor<Self>,
        cos: &Tensor<Self>,
    ) {
        let &[nh, seq, dh] = x.shape() else { panic!() };
        let &[seq_] = pos.shape() else { panic!() };
        let &[_, dh_sin] = sin.shape() else { panic!() };
        let &[_, dh_cos] = cos.shape() else { panic!() };
        assert_eq!(seq, seq_);
        assert_eq!(x.layout().strides()[2], x.dt().nbytes() as isize);

        assert_eq!(dh, dh_sin * 2);
        assert_eq!(dh, dh_cos * 2);

        let dt = Tensor::check_dt_same(&[x, sin, cos]).unwrap();
        let &[s_x_0, s_x_1, s_x_2] = x.strides() else {
            panic!()
        };
        let &[s_pos] = pos.strides() else { panic!() };
        let &[s_sin_0, s_sin_1] = sin.strides() else {
            panic!()
        };
        let &[s_cos_0, s_cos_1] = cos.strides() else {
            panic!()
        };
        assert_eq!(s_x_2, dt.nbytes() as isize);
        let scheme = Scheme {
            nh,
            seq,
            dh,
            s_x_0,
            s_x_1,
            s_pos,
            s_sin_0,
            s_sin_1,
            s_cos_0,
            s_cos_1,
            x: ptr(x).cast_mut(),
            pos: ptr(pos),
            sin: ptr(sin),
            cos: ptr(cos),
        };

        match (dt, pos.dt()) {
            (types::F16, types::U32) => scheme.calculate::<f16, u32>(),
            _ => todo!(),
        }
    }
}

struct Scheme {
    nh: usize,
    seq: usize,
    dh: usize,
    s_x_0: isize,
    s_x_1: isize,
    s_pos: isize,
    s_sin_0: isize,
    s_sin_1: isize,
    s_cos_0: isize,
    s_cos_1: isize,

    x: *mut u8,
    pos: *const u8,
    sin: *const u8,
    cos: *const u8,
}

trait Pos: Copy {
    fn pos(&self) -> usize;
}

impl Pos for u32 {
    fn pos(&self) -> usize {
        *self as _
    }
}

trait Data: Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Copy {}
impl<T> Data for T where T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy {}

impl Scheme {
    fn calculate<T: Data + Display, U: Pos>(&self) {
        let &Self {
            nh,
            seq,
            dh,
            s_sin_0,
            s_sin_1,
            s_cos_0,
            s_cos_1,
            s_pos,
            s_x_0,
            s_x_1,
            x,
            pos,
            sin,
            cos,
        } = self;

        let x = x.cast::<[T; 2]>();
        let pos = pos.cast::<U>();
        let sin = sin.cast::<T>();
        let cos = cos.cast::<T>();

        let dh = dh / 2;
        let s_x_2 = size_of::<[T; 2]>() as isize;
        for i in 0..nh * seq * dh {
            let i0 = (i / (seq * dh)) as isize;
            let i1 = ((i / dh) % seq) as isize;
            let i2 = (i % dh) as isize;

            let x = unsafe { &mut *x.byte_offset(i0 * s_x_0 + i1 * s_x_1 + i2 * s_x_2) };
            let pos = unsafe { pos.byte_offset(i1 * s_pos).read() }.pos() as isize;
            let sin = unsafe { sin.byte_offset(pos * s_sin_0 + i2 * s_sin_1).read() };
            let cos = unsafe { cos.byte_offset(pos * s_cos_0 + i2 * s_cos_1).read() };

            let [a, b] = *x;
            *x = [a * cos - b * sin, a * sin + b * cos];
        }
    }
}
