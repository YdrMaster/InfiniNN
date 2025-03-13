use crate::{ObjId, Tensor, VirtualMachine};
use std::f32::consts::PI;

pub trait RoPE: VirtualMachine {
    fn rope(
        &self,
        stack: ObjId,
        x: &mut Tensor<Self>,
        pos: &Tensor<Self>,
        sin: &Tensor<Self>,
        cos: &Tensor<Self>,
    );
}

pub enum RotaryType {
    /// 无外推
    Normal { theta: f32 },
    /// 线性内插
    PI { theta: f32, s: f32 },
    /// 非均匀频率缩放
    NtkAware { theta: f32, s: f32 },
    /// 动态适配插值比例
    DynScale { theta: f32, s: f32, a: f32 },
    /// 基于波长局部分段插值
    NtkParts {
        theta: f32,
        s: f32,
        l0: usize,
        alpha: f32,
        beta: f32,
    },
    /// YARN
    Yarn {
        theta: f32,
        s: f32,
        l0: usize,
        alpha: f32,
        beta: f32,
    },
}

impl RotaryType {
    pub fn generate(&self, nctx: usize, dh: usize) -> [Box<[f32]>; 2] {
        match *self {
            Self::Normal { theta } => generate(nctx, dh, theta, |theta, pos| theta * pos),
            Self::PI { theta, s } => generate(nctx, dh, theta, |theta, pos| theta * pos / s),
            Self::NtkAware { theta, s } => generate(nctx, dh, theta * s, |theta, pos| theta * pos),
            Self::DynScale { theta, s, a } => {
                generate(nctx, dh, theta * (a * s - a + 1.), |theta, pos| theta * pos)
            }
            Self::NtkParts {
                theta,
                s,
                l0,
                alpha,
                beta,
            } => generate(nctx, dh, theta, |theta, pos| {
                let r = l0 as f32 / (2. * PI / theta);
                let r = ((r - alpha) / (beta - alpha)).clamp(0., 1.);
                pos * ((1. - r) / s + r) * theta
            }),
            Self::Yarn {
                theta,
                s,
                l0,
                alpha,
                beta,
            } => generate(nctx, dh, theta, |theta, pos| {
                let pos = pos * (0.1 * s.ln() + 1.);
                let r = l0 as f32 / (2. * PI / theta);
                let r = ((r - alpha) / (beta - alpha)).clamp(0., 1.);
                pos * ((1. - r) / s + r) * theta
            }),
        }
    }
}

fn generate(nctx: usize, dh: usize, theta: f32, f: impl Fn(f32, f32) -> f32) -> [Box<[f32]>; 2] {
    let dh = dh / 2;
    let size = nctx * dh;
    let mut sin = vec![0.; size];
    let mut cos = vec![0.; size];
    for i in 0..size {
        let pos = (i / dh) as f32;
        let idx = (i % dh) as f32;
        let theta = theta.powf(-(idx / dh as f32));

        let (sin_, cos_) = f(theta, pos).sin_cos();

        sin[i] = sin_;
        cos[i] = cos_;
    }
    [sin.into(), cos.into()]
}
