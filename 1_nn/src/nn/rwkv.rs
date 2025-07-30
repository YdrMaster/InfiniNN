use super::{
    Context, Distribution, Embedding, Linear, NNError, Normalization, NuralNetwork, TPAction,
    TPTensor, Tensor, macros::destruct, output_head::OutputHead, weight_types::RowTPWeight,
};

#[derive(Clone)]
pub struct RWKV<T> {
    pub embedding: Embedding<T>,
    pub blks: Box<[RWKVBlock<T>]>,
    pub output_head: Option<OutputHead<T>>,
}

#[derive(Clone)]
pub struct RWKVBlock<T> {
    pub ln1: Normalization<T>,
    pub time_mix: TimeMix<T>,
    pub ln2: Normalization<T>,
    pub channel_mix: ChannelMix<T>,
    pub layer_id: usize,
}

impl<T> RWKVBlock<T> {
    #[inline]
    pub const fn new(
        ln1: Normalization<T>,
        time_mix: TimeMix<T>,
        ln2: Normalization<T>,
        channel_mix: ChannelMix<T>,
        layer_id: usize,
    ) -> Self {
        Self {
            ln1,
            time_mix,
            ln2,
            channel_mix,
            layer_id,
        }
    }

    pub fn tensor_parallel(self, dist: Distribution) -> RWKVBlock<TPTensor<T>> {
        let Self {
            ln1,
            time_mix,
            ln2,
            channel_mix,
            layer_id,
        } = self;
        RWKVBlock {
            ln1: ln1.tensor_parallel(),
            time_mix: time_mix.tensor_parallel(dist),
            ln2: ln2.tensor_parallel(),
            channel_mix: channel_mix.tensor_parallel(dist),
            layer_id,
        }
    }
}

impl<T> NuralNetwork<T> for RWKVBlock<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self {
            ln1,
            time_mix,
            ln2,
            channel_mix,
            layer_id: _,
        } = self;

        destruct!([x] = inputs);

        let residual = x.clone();
        destruct!([x] = ctx.trap("ln1", ln1, [x])?);
        destruct!([x] = ctx.trap("time-mix", time_mix, [x])?);
        let x = ctx.call("add", "add", None, [x, residual])?.remove(0);

        let residual = x.clone();
        destruct!([x] = ctx.trap("ln2", ln2, [x])?);
        destruct!([x] = ctx.trap("channel-mix", channel_mix, [x])?);
        let x = ctx.call("add", "add", None, [x, residual])?.remove(0);

        Ok((ctx, vec![x]))
    }
}

#[derive(Clone)]
pub struct TimeMix<T> {
    pub k: Linear<T>,
    pub v: Linear<T>,
    pub r: Linear<T>,
    pub time_mix_k: T,
    pub time_mix_v: T,
    pub time_mix_r: T,
    pub layer_id: usize, // 需要传递给底层算子
}

impl<T> TimeMix<T> {
    pub fn new(
        k: Linear<T>,
        v: Linear<T>,
        r: Linear<T>,
        time_mix_k: T,
        time_mix_v: T,
        time_mix_r: T,
        layer_id: usize,
    ) -> Self {
        Self {
            k,
            v,
            r,
            time_mix_k,
            time_mix_v,
            time_mix_r,
            layer_id,
        }
    }

    pub fn tensor_parallel(self, dist: Distribution) -> TimeMix<TPTensor<T>> {
        let Self {
            k,
            v,
            r,
            time_mix_k,
            time_mix_v,
            time_mix_r,
            layer_id,
        } = self;
        TimeMix {
            k: k.parallel(TPAction::new(RowTPWeight, dist)),
            v: v.parallel(TPAction::new(RowTPWeight, dist)),
            r: r.parallel(TPAction::new(RowTPWeight, dist)),
            time_mix_k: time_mix_k.into(),
            time_mix_v: time_mix_v.into(),
            time_mix_r: time_mix_r.into(),
            layer_id,
        }
    }
}

impl<T> NuralNetwork<T> for TimeMix<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self {
            k,
            v,
            r,
            time_mix_k,
            time_mix_v,
            time_mix_r,
            layer_id,
        } = self;

        destruct!([x] = inputs);

        // 加载时间混合参数
        let time_mix_k = ctx.load_external("time_mix_k", x.dt(), x.shape(), time_mix_k);
        let time_mix_v = ctx.load_external("time_mix_v", x.dt(), x.shape(), time_mix_v);
        let time_mix_r = ctx.load_external("time_mix_r", x.dt(), x.shape(), time_mix_r);

        // 线性变换
        destruct!([k_out] = ctx.trap("linear-k", k, [x.clone()])?);
        destruct!([v_out] = ctx.trap("linear-v", v, [x.clone()])?);
        destruct!([r_out] = ctx.trap("linear-r", r, [x.clone()])?);

        // RWKV时间混合 - 状态由底层算子隐式管理
        destruct!(
            [out] = ctx.call(
                "",
                "rwkv-time-mix",
                Some((layer_id as u64).into()), // 层ID作为参数传递给算子
                [x, k_out, v_out, r_out, time_mix_k, time_mix_v, time_mix_r]
            )?
        );

        Ok((ctx, vec![out]))
    }
}

#[derive(Clone)]
pub struct ChannelMix<T> {
    pub k: Linear<T>,
    pub r: Linear<T>,
    pub v: Linear<T>,
    pub time_mix_k: T,
    pub time_mix_r: T,
    pub layer_id: usize,
}

impl<T> ChannelMix<T> {
    pub fn new(
        k: Linear<T>,
        r: Linear<T>,
        v: Linear<T>,
        time_mix_k: T,
        time_mix_r: T,
        layer_id: usize,
    ) -> Self {
        Self {
            k,
            r,
            v,
            time_mix_k,
            time_mix_r,
            layer_id,
        }
    }

    pub fn tensor_parallel(self, dist: Distribution) -> ChannelMix<TPTensor<T>> {
        let Self {
            k,
            r,
            v,
            time_mix_k,
            time_mix_r,
            layer_id,
        } = self;
        ChannelMix {
            k: k.parallel(TPAction::new(RowTPWeight, dist)),
            r: r.parallel(TPAction::new(RowTPWeight, dist)),
            v: v.parallel(TPAction::new(RowTPWeight, dist)),
            time_mix_k: time_mix_k.into(),
            time_mix_r: time_mix_r.into(),
            layer_id,
        }
    }
}

impl<T> NuralNetwork<T> for ChannelMix<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self {
            k,
            r,
            v,
            time_mix_k,
            time_mix_r,
            layer_id,
        } = self;

        destruct!([x] = inputs);

        let time_mix_k = ctx.load_external("time_mix_k", x.dt(), x.shape(), time_mix_k);
        let time_mix_r = ctx.load_external("time_mix_r", x.dt(), x.shape(), time_mix_r);

        destruct!([k_out] = ctx.trap("linear-k", k, [x.clone()])?);
        destruct!([r_out] = ctx.trap("linear-r", r, [x.clone()])?);

        // RWKV通道混合 - 状态由底层算子隐式管理
        destruct!(
            [mixed] = ctx.call(
                "",
                "rwkv-channel-mix",
                Some((layer_id as u64).into()),
                [x, k_out, r_out, time_mix_k, time_mix_r]
            )?
        );

        destruct!([out] = ctx.trap("linear-v", v, [mixed])?);

        Ok((ctx, vec![out]))
    }
}

impl<T> RWKV<T> {
    pub fn new(
        embedding: Embedding<T>,
        blks: impl IntoIterator<
            Item = (
                Normalization<T>,
                TimeMix<T>,
                Normalization<T>,
                ChannelMix<T>,
            ),
        >,
        output_head: Option<OutputHead<T>>,
    ) -> Self {
        Self {
            embedding,
            blks: blks
                .into_iter()
                .enumerate()
                .map(|(layer_id, (ln1, mut time_mix, ln2, mut channel_mix))| {
                    time_mix.layer_id = layer_id;
                    channel_mix.layer_id = layer_id;
                    RWKVBlock::new(ln1, time_mix, ln2, channel_mix, layer_id)
                })
                .collect(),
            output_head,
        }
    }

    pub fn tensor_parallel(self, dist: Distribution) -> RWKV<TPTensor<T>> {
        let Self {
            embedding,
            blks,
            output_head,
        } = self;
        RWKV {
            embedding: embedding.tensor_parallel(),
            blks: blks
                .into_iter()
                .map(|blk| blk.tensor_parallel(dist))
                .collect(),
            output_head: output_head.map(|head| head.tensor_parallel()),
        }
    }
}

impl<T> NuralNetwork<T> for RWKV<T> {
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

        destruct!([x] = ctx.trap("embedding", embedding, [tokens])?);

        let x = blks.into_iter().enumerate().try_fold(x, |x, (i, blk)| {
            destruct!([x] = ctx.trap(format!("blk{i}"), blk, [x])?);
            Ok(x)
        })?;

        let x = if let Some(output_head) = output_head {
            let out_idx = inputs.next().unwrap();
            destruct!([x] = ctx.call("out-gather", "embedding", None, [x, out_idx])?);
            destruct!([x] = ctx.trap("out-norm", output_head.out_norm, [x])?);
            destruct!([x] = ctx.trap("lm-head", output_head.lm_head, [x])?);
            x
        } else {
            x
        };

        Ok((ctx, vec![x]))
    }
}
