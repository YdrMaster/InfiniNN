use super::{
    NuralNetwork, attention,
    linear::{self, Linear},
};
use crate::{
    Context, Tensor, VirtualMachine,
    nn::attention::{Attention, KVCache},
    op::{self, Add, RoPE},
    split,
};
use digit_layout::DigitLayout;
use itertools::izip;

pub struct SelfAttn {
    dt_w: DigitLayout,
    nh: usize,
    nkvh: usize,
    dh: usize,
    mask: op::AttnMask,
    attn_qkv_bias: bool,
    attn_o_bias: bool,
}

pub struct Args<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    y: Tensor<'vm, VM>,   // [n, d]
    x: Tensor<'vm, VM>,   // [n, d]
    pos: Tensor<'vm, VM>, // [n]
    n_sin: usize,
    n_cos: usize,
    reqs: Vec<Request<'vm, VM>>,
    residual: bool,
}

pub struct Request<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    k_cache: Tensor<'vm, VM>, // [k_buf, nkvh, dh]
    v_cache: Tensor<'vm, VM>, // [v_buf, nkvh, dh]
    n_seq: usize,
    pos: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Obj {
    Sin,
    Cos,
    AttnOWeight,
    AttnOBias,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Sub {
    AttnQkvLinear,
    Attn(usize),
}

pub trait Ops: RoPE + attention::Ops + Add {}
impl<VM> Ops for VM where VM: RoPE + attention::Ops + Add {}

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
            attn_qkv_bias,
            attn_o_bias,
        } = self;
        let Args {
            mut y,
            mut x,
            pos,
            n_sin,
            n_cos,
            reqs,
            residual,
        } = args;

        let dt = x.dt();
        let &[n, d] = x.shape() else { panic!() };

        let qkv = ctx.workspace(dt, &[n, (nh + nkvh + nkvh) * dh]);
        ctx.trap(
            Sub::AttnQkvLinear,
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

        let w = ctx
            .get_mapped(Obj::AttnOWeight, dt_w, &[d, nh * dh])
            .transpose(&[1, 0]);
        if attn_o_bias {
            {
                let x1 = if residual { &mut x } else { &mut y };
                {
                    let down_bias = ctx
                        .get_mapped(Obj::AttnOBias, dt_w, &[1, d])
                        .broadcast(0, n);
                    ctx.rearrange(x1, &down_bias)
                }
                ctx.mat_mul(x1, 1., &o, &w, 1.)
            }
            if residual {
                ctx.add(&mut y, &x)
            }
        } else {
            let beta = if residual { 1. } else { 0. };
            ctx.mat_mul(&mut y, beta, &o, &w, 1.)
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Obj, Request, SelfAttn, Sub};
    use crate::{Exec, Map, VirtualMachine, dev_id, nn::linear, op::AttnMask, test::TestVM};
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
            residual: true,
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
            let linear = self_attn.step_into::<linear::Linear>(Sub::AttnQkvLinear);
            linear.map_host(linear::Obj::Weight, Box::new(attn_qkv_w));
            linear.map_host(linear::Obj::Bias, Box::new(attn_qkv_b));
            self_attn.map_host(Obj::AttnOWeight, Box::new(attn_o_w));
            self_attn.map_host(Obj::AttnOBias, Box::new(attn_o_b));
            self_attn.map_host(Obj::Sin, Box::new(sin));
            self_attn.map_host(Obj::Cos, Box::new(cos));

            vm.exec(
                pid,
                DEVICE,
                &SelfAttn {
                    dt_w: DT_W,
                    nh: NH,
                    nkvh: NKVH,
                    dh: DH,
                    mask: AttnMask::None,
                    attn_qkv_bias: true,
                    attn_o_bias: true,
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
            let linear = self_attn.step_into::<linear::Linear>(Sub::AttnQkvLinear);
            linear.map_host(linear::Obj::Weight, Box::new(attn_qkv_w));
            linear.map_host(linear::Obj::Bias, Box::new(attn_qkv_b));
            self_attn.map_host(Obj::AttnOWeight, Box::new(attn_o_w));
            self_attn.map_host(Obj::Sin, Box::new(sin));
            self_attn.map_host(Obj::Cos, Box::new(cos));

            vm.exec(
                pid,
                DEVICE,
                &SelfAttn {
                    dt_w: DT_W,
                    nh: NH,
                    nkvh: NKVH,
                    dh: DH,
                    mask: AttnMask::Causal,
                    attn_qkv_bias: true,
                    attn_o_bias: false,
                },
                args(&vm),
            )
        }

        vm.unregister(pid)
    }
}
