mod activation;
mod attention;
mod cogvlm;
mod distribution;
mod embedding;
mod linear;
mod llama;
mod mamba;
mod merger;
mod mlp;
mod normalization;
mod output_head;
mod patch_embd;
mod qw2vl_mmproj;
mod rwkv;
mod transformer_blk;

use crate::{
    ctx::{Context, Tensor},
    op::OpError,
};

pub use activation::Activation;
pub use attention::{Attention, RoPE};
pub use cogvlm::CogVLM;
pub use distribution::{Distribution, TPAction, TPTensor, WeightType, weight_types};
pub use embedding::{Embedding, Table};
pub use linear::Linear;
pub use llama::LLaMA;
pub use mamba::{CausalConv1d, Mamba, MambaBlock, MambaMixer, SelectiveSSM};
pub use merger::Merger;
pub use mlp::Mlp;
pub use normalization::{Normalization, Type as NormType};
pub use output_head::OutputHead;
pub use patch_embd::PatchEmbd;
pub use qw2vl_mmproj::Qwen2VLmmproj;
pub use rwkv::{RWKV, RWKVBlock};
pub use transformer_blk::TransformerBlk;

pub trait NuralNetwork<T>: Sized {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError>;
}

#[derive(Debug)]
pub struct NNError {
    pub name: String,
    pub err: OpError,
}

pub mod macros {
    macro_rules! destruct {
        ([$( $name:ident ),+] = $iter:expr) => {
            let mut iter = $iter.into_iter();
            $( let $name = iter.next().unwrap(); )+
            assert!(iter.next().is_none());
        };
    }

    macro_rules! dims {
        ($pat:pat = $tensor:expr) => {
            let $pat = &*$tensor.shape() else {
                panic!("Ndim mismatch ( = {})", $tensor.shape().len())
            };
        };
    }

    pub(crate) use {destruct, dims};
}
