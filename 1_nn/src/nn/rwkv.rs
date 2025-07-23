use super::{
    Context, Distribution, Embedding, NNError, NuralNetwork, RWKVBlock, TPTensor, Tensor,
    macros::destruct, output_head::OutputHead,
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

#[derive(Clone)]
pub struct TimeMix<T> {
    pub k: Linear<T>,
    pub v: Linear<T>,
    pub r: Linear<T>,
    // 可能还需要时间权重参数
}

#[derive(Clone)]
pub struct ChannelMix<T> {
    pub k: Linear<T>,
    pub r: Linear<T>,
    pub v: Linear<T>,
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
        let mut state = inputs.next().unwrap(); // RWKV 状态张量

        // 词嵌入
        destruct!([x] = ctx.trap("embedding", embedding, [tokens])?);

        // RWKV 块处理 - 与 LLaMA 的关键差异
        let (x, new_state) =
            blks.into_iter()
                .enumerate()
                .try_fold((x, state), |(x, state), (i, blk)| {
                    // RWKV 块需要状态输入和输出
                    destruct!(
                        [x, layer_state] = ctx.trap(
                            format!("blk{i}"),
                            blk,
                            [x, state.slice(i)] // 传入当前层状态
                        )?
                    );
                    Ok((x, state.update_layer(i, layer_state))) // 更新状态
                })?;

        // 输出处理
        let x = if let Some(OutputHead { out_norm, lm_head }) = output_head {
            let out_idx = inputs.next().unwrap();
            destruct!([x] = ctx.call("out-gather", "embedding", None, [x, out_idx])?);
            destruct!([x] = ctx.trap("out-norm", out_norm, [x])?);
            destruct!([x] = ctx.trap("lm-head", lm_head, [x])?);
            x
        } else {
            x
        };

        Ok((ctx, vec![x, new_state])) // 返回新状态
    }
}
