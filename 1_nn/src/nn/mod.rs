mod activation;
mod attention;
mod embedding;
mod linear;
mod llama;
mod mlp;
mod normalization;
mod transformer_blk;

use crate::{
    OpError,
    ctx::{Context, Tensor},
};

pub use activation::Activation;
pub use attention::{Attention, Cache, RoPE};
pub use embedding::{Embedding, Table};
pub use linear::Linear;
pub use llama::LLaMA;
pub use mlp::Mlp;
pub use normalization::{Normalization, Type as NormType};
pub use transformer_blk::TransformerBlk;

pub trait NuralNetwork<T> {
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
