use super::{
    Context, Distribution, Embedding, NNError, Normalization, NuralNetwork, TPTensor, Tensor,
    output_head::OutputHead,
};
use crate::macros::{destruct, dims};
use crate::{Activation, Linear, TPAction, weight_types::RowTPWeight};
use arg::{Arg, Dim};
use tensor::digit_layout::DigitLayout;

#[derive(Clone)]
pub struct Mamba<T> {
    pub embedding: Embedding<T>,
    pub blks: Box<[MambaBlock<T>]>,
    pub output_head: Option<OutputHead<T>>,
}

impl<T> Mamba<T> {
    pub fn tensor_parallel(self, dist: Distribution) -> Mamba<TPTensor<T>> {
        let Self {
            embedding,
            blks,
            output_head,
        } = self;
        Mamba {
            embedding: embedding.tensor_parallel(),
            blks: blks
                .into_iter()
                .map(|blk| blk.tensor_parallel(dist))
                .collect(),
            output_head: output_head.map(OutputHead::tensor_parallel),
        }
    }
}

impl<T> NuralNetwork<T> for Mamba<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self {
            embedding,
            blks,
            output_head,
        } = self;

        let mut inputs = inputs.into_iter();
        let tokens = inputs.next().unwrap();
        let pos = inputs.next().unwrap();

        destruct!([x] = ctx.trap("embedding", embedding, [tokens])?);

        let x = blks.into_iter().enumerate().try_fold(x, |x, (i, blk)| {
            destruct!([x] = ctx.trap(format!("blk{i}"), blk, [x, pos.clone()])?);
            Ok(x)
        })?;

        let x = if let Some(OutputHead { out_norm, lm_head }) = output_head {
            let out_idx = inputs.next().unwrap();
            destruct!([x] = ctx.call("out-gather", "embedding", None, [x, out_idx])?);
            destruct!([x] = ctx.trap("out-norm", out_norm, [x])?);
            destruct!([x] = ctx.trap("lm-head", lm_head, [x])?);
            x
        } else {
            x
        };

        Ok((ctx, vec![x]))
    }
}

#[derive(Clone)]
pub struct MambaBlock<T> {
    pub mamba_norm: Normalization<T>,
    pub mamba_mixer: MambaMixer<T>,
}

#[derive(Clone)]
pub struct MambaMixer<T> {
    pub d_inner: usize,
    pub in_proj: Linear<T>,
    pub causal_conv1d: CausalConv1d<T>,
    pub act: Activation,
    pub selective_ssm: SelectiveSSM<T>,
    pub out_proj: Linear<T>,
}

impl<T> MambaBlock<T> {
    pub fn tensor_parallel(self, dist: Distribution) -> MambaBlock<TPTensor<T>> {
        let Self {
            mamba_norm,
            mamba_mixer:
                MambaMixer {
                    d_inner,
                    causal_conv1d,
                    act,
                    in_proj,
                    selective_ssm,
                    out_proj,
                },
        } = self;

        if dist.is_mono() {
            MambaBlock {
                mamba_norm: mamba_norm.tensor_parallel(),
                mamba_mixer: MambaMixer {
                    d_inner,
                    causal_conv1d: causal_conv1d.tensor_parallel(dist),
                    act,
                    in_proj: in_proj.parallel(TPAction::new(RowTPWeight, dist)),
                    selective_ssm: selective_ssm.tensor_parallel(dist),
                    out_proj: out_proj.parallel(TPAction::new(RowTPWeight, dist)),
                },
            }
        } else {
            todo!();
        }
    }
}

impl<T> NuralNetwork<T> for MambaBlock<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self {
            mamba_norm,
            mamba_mixer:
                MambaMixer {
                    d_inner,
                    causal_conv1d,
                    act,
                    in_proj,
                    selective_ssm,
                    out_proj,
                },
        } = self;
        destruct!([x, _pos] = inputs);
        dims!([_l, _d] = x);

        let residual = x.clone();
        destruct!([x] = ctx.trap("rms-norm", mamba_norm, [x])?);
        destruct!([x] = ctx.trap("in-proj", in_proj, [x])?);
        destruct!([x, gate] = x.split("split-x-gate", 1, [d_inner, d_inner].map(Dim::from))?);
        destruct!([x] = ctx.trap("causal-conv1d", causal_conv1d, [x])?);
        destruct!([x] = ctx.trap("silu", act, [x])?);

        destruct!([x] = ctx.trap("selective-ssm", selective_ssm, [x])?);
        destruct!([gate] = ctx.trap("silu", act, [gate])?);
        destruct!([x] = ctx.call("gate", "element-mul", None, [x, gate])?);
        destruct!([x] = ctx.trap("out-proj", out_proj, [x, residual])?);

        Ok((ctx, vec![x]))
    }
}

#[derive(Clone)]
pub struct CausalConv1d<T> {
    pub dt: DigitLayout,
    pub casual_conv1d_w: T,
    pub casual_conv1d_b: T,
    pub d_kernel: usize,
    pub d_inner: usize,
    groups: usize,
    padding: usize,
}

impl<T> CausalConv1d<T> {
    #[inline]
    pub const fn new(
        dt: DigitLayout,
        casual_conv1d_w: T,
        casual_conv1d_b: T,
        d_kernel: usize,
        d_inner: usize,
    ) -> Self {
        Self {
            dt,
            casual_conv1d_w,
            casual_conv1d_b,
            d_kernel,
            d_inner,
            groups: d_inner,
            padding: d_kernel - 1,
        }
    }
    pub fn tensor_parallel(self, dist: Distribution) -> CausalConv1d<TPTensor<T>> {
        let Self {
            dt,
            casual_conv1d_w,
            casual_conv1d_b,
            d_kernel,
            d_inner,
            padding,
            groups,
        } = self;

        if dist.is_mono() {
            CausalConv1d {
                dt,
                casual_conv1d_w: TPTensor::from(casual_conv1d_w),
                casual_conv1d_b: TPTensor::from(casual_conv1d_b),
                d_kernel,
                d_inner,
                padding,
                groups,
            }
        } else {
            todo!();
        }
    }
}

impl<T> NuralNetwork<T> for CausalConv1d<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self {
            dt,
            casual_conv1d_w: w,
            casual_conv1d_b: b,
            d_kernel,
            d_inner,
            padding,
            groups,
        } = self;
        destruct!([x] = inputs);
        dims!([_l, _d_in] = x);
        let [kernel_size, inner_size] = [d_kernel, d_inner].map(Dim::from);

        let w = ctx.load_external(
            "causal_conv1d_weight",
            dt,
            [inner_size.clone(), kernel_size],
            w,
        );
        let b = ctx.load_external("causal_conv1d_bias", dt, [inner_size], b);
        let arg = Arg::arr([groups, padding].map(|x| Arg::from(x as u64)));
        let tensors = ctx.call("", "mamba-causal-conv1d", Some(arg), [x, w, b]);
        destruct!([out] = tensors);

        Ok((ctx, out))
    }
}

#[derive(Clone)]
pub struct SelectiveSSM<T> {
    pub dt: DigitLayout,
    pub d_state: usize,
    pub dt_rank: usize,
    pub x_proj: Linear<T>,
    pub dt_proj: Linear<T>,
    pub a: T,
    pub d: T,
}

impl<T> SelectiveSSM<T> {
    pub fn tensor_parallel(self, dist: Distribution) -> SelectiveSSM<TPTensor<T>> {
        let Self {
            dt,
            d_state,
            dt_rank,
            x_proj,
            dt_proj,
            a,
            d,
        } = self;

        if dist.is_mono() {
            SelectiveSSM {
                dt,
                d_state,
                dt_rank,
                x_proj: x_proj.parallel(TPAction::new(RowTPWeight, dist)),
                dt_proj: dt_proj.parallel(TPAction::new(RowTPWeight, dist)),
                a: TPTensor::from(a),
                d: TPTensor::from(d),
            }
        } else {
            todo!();
        }
    }
}

impl<T> NuralNetwork<T> for SelectiveSSM<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self {
            dt,
            d_state,
            dt_rank,
            x_proj,
            dt_proj,
            a,
            d,
        } = self;
        destruct!([x] = inputs);
        dims!([_l, d_in] = x);

        destruct!([dt_b_c] = ctx.trap("x-proj", x_proj, [x.clone()])?);
        destruct!(
            [delta, b, c] = dt_b_c.split(
                "split-dt-b-c",
                1,
                [dt_rank, d_state, d_state].map(Dim::from)
            )?
        );
        destruct!([delta] = ctx.trap("dt-proj", dt_proj, [delta])?);
        let a = ctx.load_external("ssm_a", dt, [d_in.clone(), Dim::from(d_state)], a);
        let d = ctx.load_external("ssm_d", dt, [d_in.clone()], d);
        let out = ctx.call("", "mamba-selective-scan", None, [x, delta, a, b, c, d])?;

        Ok((ctx, out))
    }
}
