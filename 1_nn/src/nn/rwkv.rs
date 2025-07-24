use super::{
    Context, Distribution, Embedding, Linear, NNError, Normalization, NuralNetwork, TPAction,
    TPTensor, Tensor, macros::destruct, output_head::OutputHead, rwkv_state::RWKVState,
    weight_types::RowTPWeight,
};

#[derive(Clone)]
pub struct RWKV<T> {
    pub embedding: Embedding<T>,            // 词嵌入层（与LLaMA相同）
    pub blks: Box<[RWKVBlock<T>]>,          // RWKV块序列（替代TransformerBlk）
    pub output_head: Option<OutputHead<T>>, // 输出头（与LLaMA相同）
}

#[derive(Clone)]
pub struct RWKVBlock<T> {
    pub ln1: Normalization<T>,      // 第一个归一化层
    pub time_mix: TimeMix<T>,       // 时间混合（注意力机制的替代）
    pub ln2: Normalization<T>,      // 第二个归一化层
    pub channel_mix: ChannelMix<T>, // 通道混合（FFN的替代）
}

impl<T> RWKVBlock<T> {
    #[inline]
    pub const fn new(
        ln1: Normalization<T>,
        time_mix: TimeMix<T>,
        ln2: Normalization<T>,
        channel_mix: ChannelMix<T>,
    ) -> Self {
        Self {
            ln1,
            time_mix,
            ln2,
            channel_mix,
        }
    }

    pub fn tensor_parallel(self, dist: Distribution) -> RWKVBlock<TPTensor<T>> {
        let Self {
            ln1,
            time_mix,
            ln2,
            channel_mix,
        } = self;
        RWKVBlock {
            ln1: ln1.tensor_parallel(),
            time_mix: time_mix.tensor_parallel(dist),
            ln2: ln2.tensor_parallel(),
            channel_mix: channel_mix.tensor_parallel(dist),
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
        } = self;

        destruct!([x, state] = inputs);
        let residual = x.clone();

        destruct!([x] = ctx.trap("ln1", ln1, [x])?);
        destruct!([x, new_state] = ctx.trap("time-mix", time_mix, [x, state])?);
        let x = ctx.call("add", "add", None, [x, residual])?.remove(0);

        let residual = x.clone();
        destruct!([x] = ctx.trap("ln2", ln2, [x])?);
        destruct!([x] = ctx.trap("channel-mix", channel_mix, [x, new_state.clone()])?);
        let x = ctx.call("add", "add", None, [x, residual])?.remove(0);

        Ok((ctx, vec![x, new_state]))
    }
}

#[derive(Clone)]
pub struct TimeMix<T> {
    pub k: Linear<T>,
    pub v: Linear<T>,
    pub r: Linear<T>,
}

impl<T> TimeMix<T> {
    pub fn tensor_parallel(self, dist: Distribution) -> TimeMix<TPTensor<T>> {
        let Self { k, v, r } = self;
        TimeMix {
            k: k.parallel(TPAction::new(RowTPWeight, dist)),
            v: v.parallel(TPAction::new(RowTPWeight, dist)),
            r: r.parallel(TPAction::new(RowTPWeight, dist)),
        }
    }
}

impl<T> NuralNetwork<T> for TimeMix<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self { k, v, r } = self;

        destruct!([x, state] = inputs);

        // 混合当前输入 x 和上一个状态 state
        let xk = ctx
            .call("mix_k", "mix", None, [x.clone(), state.clone()])?
            .remove(0);
        let xv = ctx
            .call("mix_v", "mix", None, [x.clone(), state.clone()])?
            .remove(0);
        let xr = ctx
            .call("mix_r", "mix", None, [x.clone(), state])?
            .remove(0);

        // 线性层：k, v, r
        destruct!([_k_out] = ctx.trap("linear-k", k, [xk])?);
        destruct!([v_out] = ctx.trap("linear-v", v, [xv])?);
        destruct!([r_out] = ctx.trap("linear-r", r, [xr])?);

        // 计算 gate: sigmoid(r) * v
        let r_act = ctx.call("sigmoid", "sigmoid", None, [r_out])?.remove(0);
        let out = ctx.call("mul", "mul", None, [r_act, v_out])?.remove(0);

        // 返回输出和更新后的状态（即当前 x）
        Ok((ctx, vec![out, x]))
    }
}

#[derive(Clone)]
pub struct ChannelMix<T> {
    pub k: Linear<T>,
    pub r: Linear<T>,
    pub v: Linear<T>,
}

impl<T> ChannelMix<T> {
    pub fn tensor_parallel(self, dist: Distribution) -> ChannelMix<TPTensor<T>> {
        let Self { k, r, v } = self;
        ChannelMix {
            k: k.parallel(TPAction::new(RowTPWeight, dist)),
            r: r.parallel(TPAction::new(RowTPWeight, dist)),
            v: v.parallel(TPAction::new(RowTPWeight, dist)),
        }
    }
}

impl<T> NuralNetwork<T> for ChannelMix<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self { k, r, v } = self;

        destruct!([x, state] = inputs);

        destruct!([r_out] = ctx.trap("linear-r", r, [x.clone()])?);
        destruct!([k_out] = ctx.trap("linear-k", k, [x])?);

        let r_act = ctx.call("sigmoid", "sigmoid", None, [r_out])?.remove(0);
        let gated = ctx.call("mul", "mul", None, [r_act, k_out])?.remove(0);

        destruct!([out] = ctx.trap("linear-v", v, [gated])?);

        Ok((ctx, vec![out, state]))
    }
}

impl<T> RWKV<T> {
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
            output_head: output_head.map(OutputHead::tensor_parallel),
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
        let state_tensor = inputs.next().unwrap();

        let n_layers = blks.len();
        let state = RWKVState::from_tensor(state_tensor, n_layers)?;

        destruct!([x] = ctx.trap("embedding", embedding, [tokens])?);

        let (x, new_state) =
            blks.into_iter()
                .enumerate()
                .try_fold((x, state), |(x, state), (i, blk)| {
                    let (layer_state, builder) = state.into_layer(i);
                    destruct!(
                        [x, new_layer_state] =
                            ctx.trap(format!("blk{i}"), blk, [x, layer_state])?
                    );
                    Ok((x, builder.with_layer(new_layer_state)))
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

        let final_state = new_state.to_tensor(&mut ctx)?;
        Ok((ctx, vec![x, final_state]))
    }
}
