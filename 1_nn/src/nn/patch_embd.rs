use super::{Context, NNError, NuralNetwork, Tensor, macros::destruct};

#[allow(dead_code)]
#[derive(Clone)]
pub struct PatchEmbd {}

impl<T> NuralNetwork<T> for PatchEmbd {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        destruct!([x, w, b] = inputs);

        let tensors = ctx
            .call("", "conv", None, [x.clone(), w.clone(), b.clone()])
            .unwrap();
        destruct!([patch_embd] = tensors);
        let tensors = ctx.call("", "conv", None, [x, w, b]).unwrap();
        destruct!([patch_embd1] = tensors);
        let output = ctx
            .call("", "add", None, [patch_embd, patch_embd1])
            .unwrap();

        Ok((ctx, output))
    }
}
