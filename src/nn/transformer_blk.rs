use super::{
    NuralNetwork, WeightBiasData,
    mlp::{self, Activation, Mlp},
    normalization::{self as norm, Normalization},
    self_attn::{self, Request, SelfAttn},
};
use crate::{Context, Id, Mapping, Tensor, VirtualMachine, op::AttnMask};
use digit_layout::DigitLayout;

pub struct TransformerBlk {
    pub norm: Normalization,
    pub self_attn: SelfAttn,
    pub mlp: Mlp,
}

impl TransformerBlk {
    pub fn llama(
        dt_norm: DigitLayout,
        dt_w: DigitLayout,
        nh: usize,
        nkvh: usize,
        dh: usize,
        di: usize,
        epsilon: f32,
    ) -> Self {
        Self {
            norm: Normalization {
                ty: norm::Type::RmsNorm { epsilon },
                dt_w: dt_norm,
            },
            self_attn: SelfAttn {
                dt_w,
                nh,
                nkvh,
                dh,
                mask: AttnMask::Causal,
                qkv_bias: false,
                o_bias: false,
            },
            mlp: Mlp {
                act: Activation::SwiGLU,
                dt_w,
                di,
                up_bias: false,
                down_bias: false,
            },
        }
    }

    pub fn qwen2(
        dt_norm: DigitLayout,
        dt_w: DigitLayout,
        nh: usize,
        nkvh: usize,
        dh: usize,
        di: usize,
        epsilon: f32,
    ) -> Self {
        let mut llama = Self::llama(dt_norm, dt_w, nh, nkvh, dh, di, epsilon);
        llama.self_attn.qkv_bias = true;
        llama
    }

    pub fn gpt2(
        dt_norm: DigitLayout,
        dt_w: DigitLayout,
        nh: usize,
        nkvh: usize,
        dh: usize,
        di: usize,
    ) -> Self {
        Self {
            norm: Normalization {
                ty: norm::Type::LayerNorm,
                dt_w: dt_norm,
            },
            self_attn: SelfAttn {
                dt_w,
                nh,
                nkvh,
                dh,
                mask: AttnMask::Causal,
                qkv_bias: true,
                o_bias: true,
            },
            mlp: Mlp {
                act: Activation::GeLU,
                dt_w,
                di,
                up_bias: true,
                down_bias: true,
            },
        }
    }
}

pub struct Args<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    pub embd: Tensor<'vm, VM>, // [n, d]
    pub pos: Tensor<'vm, VM>,  // [d]
    pub n_sin: usize,
    pub n_cos: usize,
    pub reqs: Vec<Request<'vm, VM>>,
}

pub struct Data {
    pre_norm: WeightBiasData,
    self_attn: self_attn::Data,
    post_norm: WeightBiasData,
    mlp: mlp::Data,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Sub {
    PreNorm,
    SelfAttn,
    PostNorm,
    Mlp,
}

impl Id for Sub {
    fn name(&self) -> &str {
        match self {
            Sub::PreNorm => "pre-norm",
            Sub::SelfAttn => "self-attn",
            Sub::PostNorm => "post-norm",
            Sub::Mlp => "mlp",
        }
    }
}

pub trait Ops: norm::Ops + self_attn::Ops + mlp::Ops {}
impl<VM> Ops for VM where VM: norm::Ops + self_attn::Ops + mlp::Ops + ?Sized {}

impl<VM> NuralNetwork<VM> for TransformerBlk
where
    VM: VirtualMachine + ?Sized + Ops,
{
    type Args<'vm>
        = Args<'vm, VM>
    where
        VM: 'vm;
    type Data = Data;
    type Obj = ();
    type Sub = Sub;

    fn init(data: Self::Data, mut mapping: Mapping<VM, Self>) {
        let Self::Data {
            pre_norm,
            self_attn,
            post_norm,
            mlp,
        } = data;
        mapping
            .trap::<Normalization>(Sub::PreNorm, pre_norm)
            .trap::<SelfAttn>(Sub::SelfAttn, self_attn)
            .trap::<Normalization>(Sub::PostNorm, post_norm)
            .trap::<Mlp>(Sub::Mlp, mlp);
    }

    fn forward(&self, args: Self::Args<'_>, mut ctx: Context<VM, Self>) {
        let Self {
            norm,
            self_attn,
            mlp,
        } = self;
        let Args {
            embd: x,
            pos,
            n_sin,
            n_cos,
            reqs,
        } = args;

        let x1 = ctx.workspace(x.dt(), x.shape());

        ctx.trap(
            Sub::PreNorm,
            norm,
            norm::Args {
                y: x1.clone(),
                x: x.clone(),
            },
        )
        .trap(
            Sub::SelfAttn,
            self_attn,
            self_attn::Args {
                y: x.clone(),
                x: x1.clone(),
                pos,
                n_sin,
                n_cos,
                reqs,
            },
        )
        .trap(
            Sub::PostNorm,
            norm,
            norm::Args {
                y: x1.clone(),
                x: x.clone(),
            },
        )
        .trap(
            Sub::Mlp,
            mlp,
            mlp::Args {
                y: x,
                x: x1,
                scale: 1.,
                residual: true,
            },
        );
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Data, TransformerBlk};
    use crate::{
        VirtualMachine, VirtualMachineExt, dev_id,
        nn::{
            WeightBiasData, mlp,
            self_attn::{self, Request},
        },
        test::{TestVM, test_data},
    };
    use digit_layout::{DigitLayout, types as ty};

    #[test]
    fn test() {
        const DEVICE: dev_id = 0;

        const DT_W: DigitLayout = ty::F16;
        const DT_NORM: DigitLayout = ty::F32;
        const NH: usize = 64;
        const NKVH: usize = 8;
        const DH: usize = 128;
        const N: usize = 11;
        const D: usize = 2048;
        const MAX_CTX: usize = 4096;
        const DI: usize = 11008;

        let vm = TestVM::default();
        let pid = vm.register("transformer");

        vm.init::<TransformerBlk>(
            pid,
            DEVICE,
            Data {
                pre_norm: WeightBiasData {
                    weight: test_data(DT_NORM, &[D]),
                    bias: None,
                },
                self_attn: self_attn::Data {
                    qkv: WeightBiasData {
                        weight: test_data(DT_W, &[(NH + NKVH + NKVH) * DH, D]),
                        bias: None,
                    },
                    sin: test_data(DT_W, &[MAX_CTX, DH / 2]),
                    cos: test_data(DT_W, &[MAX_CTX, DH / 2]),
                    output: WeightBiasData {
                        weight: test_data(DT_W, &[D, NH * DH]),
                        bias: None,
                    },
                },
                post_norm: WeightBiasData {
                    weight: test_data(DT_NORM, &[D]),
                    bias: None,
                },
                mlp: mlp::Data {
                    up: WeightBiasData {
                        weight: test_data(DT_W, &[D * DI * 2]),
                        bias: None,
                    },
                    down: WeightBiasData {
                        weight: test_data(DT_W, &[DI * D]),
                        bias: None,
                    },
                },
            },
        )
        .forward(
            pid,
            DEVICE,
            &TransformerBlk::llama(DT_NORM, DT_W, NH, NKVH, DH, DI, 1e-5),
            Args {
                embd: vm.workspace(DT_W, &[N, D]),
                pos: vm.workspace(ty::U32, &[N]),
                n_sin: MAX_CTX,
                n_cos: MAX_CTX,
                reqs: vec![
                    Request {
                        k_cache: vm.workspace(ty::F16, &[MAX_CTX, NKVH, DH]),
                        v_cache: vm.workspace(ty::F16, &[MAX_CTX, NKVH, DH]),
                        n_seq: 7,
                        pos: 20,
                    },
                    Request {
                        k_cache: vm.workspace(ty::F16, &[MAX_CTX, NKVH, DH]),
                        v_cache: vm.workspace(ty::F16, &[MAX_CTX, NKVH, DH]),
                        n_seq: 1,
                        pos: 30,
                    },
                    Request {
                        k_cache: vm.workspace(ty::F16, &[MAX_CTX, NKVH, DH]),
                        v_cache: vm.workspace(ty::F16, &[MAX_CTX, NKVH, DH]),
                        n_seq: 3,
                        pos: 40,
                    },
                ],
            },
        );

        vm.unregister(pid)
    }
}
