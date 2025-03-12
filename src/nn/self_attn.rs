use super::{
    NuralNetwork, attention,
    linear::{self, Linear},
};
use crate::{
    Context, Id, Tensor, VirtualMachine,
    nn::{
        attention::{Attention, KVCache},
        linear_residual::{self, LinearResidual},
    },
    op::{AttnMask, RoPE},
    split,
};
use digit_layout::DigitLayout;
use itertools::izip;

pub struct SelfAttn {
    pub dt_w: DigitLayout,
    pub nh: usize,
    pub nkvh: usize,
    pub dh: usize,
    pub mask: AttnMask,
    pub qkv_bias: bool,
    pub o_bias: bool,
}

pub struct Args<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    pub y: Tensor<'vm, VM>,   // [n, d]
    pub x: Tensor<'vm, VM>,   // [n, d]
    pub pos: Tensor<'vm, VM>, // [n]
    pub n_sin: usize,
    pub n_cos: usize,
    pub reqs: Vec<Request<'vm, VM>>,
}

pub struct Request<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    pub k_cache: Tensor<'vm, VM>, // [k_buf, nkvh, dh]
    pub v_cache: Tensor<'vm, VM>, // [v_buf, nkvh, dh]
    pub n_seq: usize,
    pub pos: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Obj {
    Sin,
    Cos,
}

impl Id for Obj {
    fn name(&self) -> &str {
        match self {
            Self::Sin => "sin",
            Self::Cos => "cos",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Sub {
    QkvLinear,
    Attn(usize),
    OutputLinear,
}

impl Id for Sub {
    fn name(&self) -> &str {
        match self {
            Self::QkvLinear => "qkv",
            Self::Attn(_) => "attn",
            Self::OutputLinear => "output",
        }
    }

    fn idx(&self) -> Option<usize> {
        match self {
            &Self::Attn(i) => Some(i),
            Self::QkvLinear | Sub::OutputLinear => None,
        }
    }
}

pub trait Ops: RoPE + attention::Ops + linear_residual::Ops {}
impl<VM> Ops for VM where VM: RoPE + attention::Ops + linear_residual::Ops + ?Sized {}

impl<VM> NuralNetwork<VM> for SelfAttn
where
    VM: VirtualMachine + ?Sized + Ops,
{
    type Args<'vm>
        = Args<'vm, VM>
    where
        VM: 'vm;
    type Obj = Obj;
    type Sub = Sub;

    fn launch(&self, args: Self::Args<'_>, mut ctx: Context<VM, Self>) {
        let &Self {
            dt_w,
            nh,
            nkvh,
            dh,
            mask,
            qkv_bias: attn_qkv_bias,
            o_bias: attn_o_bias,
        } = self;
        let Args {
            y,
            x,
            pos,
            n_sin,
            n_cos,
            reqs,
        } = args;

        let dt = x.dt();
        let &[n, _] = x.shape() else { panic!() };

        let qkv = ctx.workspace(dt, &[n, (nh + nkvh + nkvh) * dh]);
        ctx.trap(
            Sub::QkvLinear,
            &Linear {
                dt_w,
                bias: attn_qkv_bias,
            },
            linear::Args {
                y: qkv.clone(),
                x: x.clone(),
            },
        );

        let qkv = qkv.tile(1, &[nh + nkvh + nkvh, dh]);
        split!(qkv => q, k, v; [nh, nkvh, nkvh] @ 1);

        let o = q.clone().merge(1, 2).unwrap();
        let mut q = q.transpose(&[1, 0]);
        let mut k = k.transpose(&[1, 0]);
        let v = v.transpose(&[1, 0]);
        {
            let sin = ctx.get_mapped(Obj::Sin, dt, &[n_sin, dh / 2]);
            let cos = ctx.get_mapped(Obj::Cos, dt, &[n_cos, dh / 2]);
            ctx.rope(&mut q, &pos, &sin, &cos);
            ctx.rope(&mut k, &pos, &sin, &cos);
        }

        let split = reqs.iter().map(|req| req.n_seq).collect::<Vec<_>>();
        let q = q.split(1, &split);
        let k = k.split(1, &split);
        let v = v.split(1, &split);

        for (i, (req, q, k, v)) in izip!(reqs, q, k, v).enumerate() {
            let o = q.clone();
            let Request {
                k_cache,
                v_cache,
                pos,
                ..
            } = req;
            ctx.trap(
                Sub::Attn(i),
                &Attention { mask },
                attention::Args {
                    q,
                    k,
                    v,
                    o,
                    cache: Some(KVCache {
                        k_cache,
                        v_cache,
                        pos,
                    }),
                },
            )
        }

        ctx.trap(
            Sub::OutputLinear,
            &LinearResidual {
                dt_w,
                bias: attn_o_bias,
                scale: 1.,
                residual: true,
            },
            linear_residual::Args { y, x: o, y_: x },
        )
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Obj, Request, SelfAttn, Sub};
    use crate::{
        Exec, Map, VirtualMachine, dev_id,
        nn::{WeightBias, linear::Linear, linear_residual::LinearResidual},
        op::AttnMask,
        test::TestVM,
    };
    use digit_layout::{DigitLayout, types as ty};

    const ARCH: &str = "self-attn";
    const DEVICE: dev_id = 0;
    const DT_W: DigitLayout = ty::F16;
    const NH: usize = 64;
    const NKVH: usize = 8;
    const DH: usize = 128;
    const N: usize = 11;
    const D: usize = 2048;
    const MAX_CTX: usize = 4096;

    fn args(vm: &TestVM) -> Args<TestVM> {
        Args {
            y: vm.workspace(Some(DEVICE), ty::F16, &[N, D]),
            x: vm.workspace(Some(DEVICE), ty::F16, &[N, D]),
            pos: vm.workspace(Some(DEVICE), ty::U32, &[N]),
            n_sin: MAX_CTX,
            n_cos: MAX_CTX,
            reqs: vec![
                Request {
                    k_cache: vm.workspace(Some(DEVICE), ty::F16, &[MAX_CTX, NKVH, DH]),
                    v_cache: vm.workspace(Some(DEVICE), ty::F16, &[MAX_CTX, NKVH, DH]),
                    n_seq: 7,
                    pos: 20,
                },
                Request {
                    k_cache: vm.workspace(Some(DEVICE), ty::F16, &[MAX_CTX, NKVH, DH]),
                    v_cache: vm.workspace(Some(DEVICE), ty::F16, &[MAX_CTX, NKVH, DH]),
                    n_seq: 1,
                    pos: 30,
                },
                Request {
                    k_cache: vm.workspace(Some(DEVICE), ty::F16, &[MAX_CTX, NKVH, DH]),
                    v_cache: vm.workspace(Some(DEVICE), ty::F16, &[MAX_CTX, NKVH, DH]),
                    n_seq: 3,
                    pos: 40,
                },
            ],
        }
    }

    #[test]
    fn test_clip() {
        let vm = TestVM::default();
        let pid = vm.register(ARCH);

        {
            let attn_qkv_w = vec![0u8; (NH + NKVH + NKVH) * DH * D * 2];
            let attn_qkv_b = vec![0u8; (NH + NKVH + NKVH) * DH * 2];
            let sin = vec![0u8; MAX_CTX * DH / 2 * 2];
            let cos = vec![0u8; MAX_CTX * DH / 2 * 2];
            let attn_o_w = vec![0u8; D * NH * DH * 2];
            let attn_o_b = vec![0u8; D * 2];

            let self_attn = vm.map::<SelfAttn>(pid, DEVICE);
            self_attn.map_host(Obj::Sin, Box::new(sin));
            self_attn.map_host(Obj::Cos, Box::new(cos));

            let qkv = self_attn.step_into::<Linear>(Sub::QkvLinear);
            qkv.map_host(WeightBias::Weight, Box::new(attn_qkv_w));
            qkv.map_host(WeightBias::Bias, Box::new(attn_qkv_b));

            let output = self_attn.step_into::<LinearResidual>(Sub::OutputLinear);
            output.map_host(WeightBias::Weight, Box::new(attn_o_w));
            output.map_host(WeightBias::Bias, Box::new(attn_o_b));

            vm.exec(
                pid,
                DEVICE,
                &SelfAttn {
                    dt_w: DT_W,
                    nh: NH,
                    nkvh: NKVH,
                    dh: DH,
                    mask: AttnMask::None,
                    qkv_bias: true,
                    o_bias: true,
                },
                args(&vm),
            )
        }

        vm.unregister(pid)
    }

    #[test]
    fn test_qwen() {
        let vm = TestVM::default();
        let pid = vm.register(ARCH);

        {
            let attn_qkv_w = vec![0u8; (NH + NKVH + NKVH) * DH * D * 2];
            let attn_qkv_b = vec![0u8; (NH + NKVH + NKVH) * DH * 2];
            let sin = vec![0u8; MAX_CTX * DH / 2 * 2];
            let cos = vec![0u8; MAX_CTX * DH / 2 * 2];
            let attn_o_w = vec![0u8; D * NH * DH * 2];

            let self_attn = vm.map::<SelfAttn>(pid, DEVICE);
            self_attn.map_host(Obj::Sin, Box::new(sin));
            self_attn.map_host(Obj::Cos, Box::new(cos));

            let qkv = self_attn.step_into::<Linear>(Sub::QkvLinear);
            qkv.map_host(WeightBias::Weight, Box::new(attn_qkv_w));
            qkv.map_host(WeightBias::Bias, Box::new(attn_qkv_b));

            let output = self_attn.step_into::<LinearResidual>(Sub::OutputLinear);
            output.map_host(WeightBias::Weight, Box::new(attn_o_w));

            vm.exec(
                pid,
                DEVICE,
                &SelfAttn {
                    dt_w: DT_W,
                    nh: NH,
                    nkvh: NKVH,
                    dh: DH,
                    mask: AttnMask::Causal,
                    qkv_bias: true,
                    o_bias: false,
                },
                args(&vm),
            )
        }

        vm.unregister(pid)
    }
}
