use super::{
    NuralNetwork,
    mlp::{self, Mlp},
    normalization::{self as norm, Normalization},
    self_attn::{self, Request, SelfAttn},
};
use crate::{Context, Id, Tensor, VirtualMachine};

pub struct TransformerBlk {
    pub norm: Normalization,
    pub self_attn: SelfAttn,
    pub mlp: Mlp,
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
    type Obj = ();
    type Sub = Sub;

    fn launch(&self, args: Self::Args<'_>, mut ctx: Context<VM, Self>) {
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
        );

        ctx.trap(
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
        );

        ctx.trap(
            Sub::PostNorm,
            norm,
            norm::Args {
                y: x1.clone(),
                x: x.clone(),
            },
        );

        ctx.trap(
            Sub::Mlp,
            mlp,
            mlp::Args {
                y: x,
                x: x1,
                scale: 1.,
                residual: true,
            },
        )
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Sub, TransformerBlk};
    use crate::{
        Exec, Map, VirtualMachine,
        nn::{
            WeightBias,
            linear::Linear,
            linear_residual::LinearResidual,
            mlp::{self, Activation, Mlp},
            normalization::{Normalization, Type as NormType},
            self_attn::{self, Obj, SelfAttn},
        },
        op::AttnMask,
        test::TestVM,
    };
    use digit_layout::{DigitLayout, types as ty};

    #[test]
    fn test() {
        const DT_W: DigitLayout = ty::F16;
        const DT_NORM: DigitLayout = ty::F32;
        const NH: usize = 64;
        const NKVH: usize = 8;
        const DH: usize = 128;
        const N: usize = 11;
        const D: usize = 2048;
        const MAX_CTX: usize = 4096;
        const DI: usize = 11008;

        const NORM: NormType = NormType::RmsNorm { epsilon: 1e-5 };
        const ATTN_QKV_BIAS: bool = false;
        const ATTN_O_BIAS: bool = false;
        const MASK: AttnMask = AttnMask::None;
        const ACT: Activation = Activation::GeLU;

        const MLP_UP_BIAS: bool = false;
        const MLP_DOWN_BIAS: bool = false;

        let vm = TestVM::default();
        let pid = vm.register("transformer");
        let device = 0;
        // 参数加载
        let transformer = vm.map::<TransformerBlk>(pid, device);

        // 1. 预归一化
        let pre_norm = transformer.step_into::<Normalization>(Sub::PreNorm);

        //dt = F32
        let w = vec![0u8; D * DT_NORM.nbytes()];
        pre_norm.map_host(WeightBias::Weight, Box::new(w));

        // 2. 自注意力
        let self_attn = transformer.step_into::<SelfAttn>(Sub::SelfAttn);

        let sin = vec![0u8; MAX_CTX * DH / 2 * 2];
        let cos = vec![0u8; MAX_CTX * DH / 2 * 2];
        let attn_qkv_w = vec![0u8; (NH + NKVH + NKVH) * DH * D * 2];
        let attn_o_w = vec![0u8; D * NH * DH * 2];

        self_attn.map_host(Obj::Sin, Box::new(sin));
        self_attn.map_host(Obj::Cos, Box::new(cos));

        let qkv = self_attn.step_into::<Linear>(self_attn::Sub::QkvLinear);
        qkv.map_host(WeightBias::Weight, Box::new(attn_qkv_w));

        let output = self_attn.step_into::<LinearResidual>(self_attn::Sub::OutputLinear);
        output.map_host(WeightBias::Weight, Box::new(attn_o_w));

        // 3. 后归一化
        let post_norm = transformer.step_into::<Normalization>(Sub::PostNorm);

        //dt = F32
        let w = vec![0u8; D * DT_NORM.nbytes()];
        post_norm.map_host(WeightBias::Weight, Box::new(w));

        // 4. MLP处理
        let mlp = transformer.step_into::<Mlp>(Sub::Mlp);
        // wup: D x DI, 每个元素 2 字节(F16)
        // 对于 GeLU 激活函数，上投影维度就是 DI
        let wup = vec![0u8; D * DI * DT_W.nbytes()];
        // wdown: DI x D, 每个元素 2 字节(F16)
        let wdown = vec![0u8; DI * D * DT_W.nbytes()];

        mlp.step_into::<Linear>(mlp::Sub::UpLinear)
            .map_host(WeightBias::Weight, Box::new(wup));
        mlp.step_into::<Linear>(mlp::Sub::DownLinear)
            .map_host(WeightBias::Weight, Box::new(wdown));

        // 创建输入输出张量
        let embd = vm.workspace(Some(device), DT_W, &[N, D]);
        let pos = vm.workspace(Some(device), ty::U32, &[N]);

        // 执行 transformer
        vm.exec(
            pid,
            device,
            &TransformerBlk {
                norm: Normalization {
                    ty: NORM,
                    dt_w: DT_NORM,
                },
                self_attn: SelfAttn {
                    dt_w: DT_W,
                    nh: NH,
                    nkvh: NKVH,
                    dh: DH,
                    mask: MASK,
                    qkv_bias: ATTN_QKV_BIAS,
                    o_bias: ATTN_O_BIAS,
                },
                mlp: Mlp {
                    act: ACT,
                    dt_w: DT_W,
                    di: DI,
                    up_bias: MLP_UP_BIAS,
                    down_bias: MLP_DOWN_BIAS,
                },
            },
            Args {
                embd,
                pos,
                n_sin: MAX_CTX,
                n_cos: MAX_CTX,
                reqs: vec![
                    self_attn::Request {
                        k_cache: vm.workspace(Some(device), ty::F16, &[MAX_CTX, NKVH, DH]),
                        v_cache: vm.workspace(Some(device), ty::F16, &[MAX_CTX, NKVH, DH]),
                        n_seq: 7,
                        pos: 20,
                    },
                    self_attn::Request {
                        k_cache: vm.workspace(Some(device), ty::F16, &[MAX_CTX, NKVH, DH]),
                        v_cache: vm.workspace(Some(device), ty::F16, &[MAX_CTX, NKVH, DH]),
                        n_seq: 1,
                        pos: 30,
                    },
                    self_attn::Request {
                        k_cache: vm.workspace(Some(device), ty::F16, &[MAX_CTX, NKVH, DH]),
                        v_cache: vm.workspace(Some(device), ty::F16, &[MAX_CTX, NKVH, DH]),
                        n_seq: 3,
                        pos: 40,
                    },
                ],
            },
        );

        vm.unregister(pid);
    }
}
