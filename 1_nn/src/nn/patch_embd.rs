use super::{Context, Distribution, NNError, NuralNetwork, TPTensor, Tensor, macros::destruct};
use crate::macros::dims;
use arg::Dim;
use tensor::digit_layout::DigitLayout;

#[derive(Clone)]
pub struct PatchEmbd<T> {
    pub dt: DigitLayout,
    pub shape: [usize; 4],
    pub patch_embd: T,
    pub patch_embd1: T,
}

impl<T> PatchEmbd<T> {
    pub fn tensor_parallel(self, dist: Distribution) -> PatchEmbd<TPTensor<T>> {
        let Self {
            dt,
            shape,
            patch_embd,
            patch_embd1,
        } = self;

        if dist.is_mono() {
            PatchEmbd {
                dt,
                shape,
                patch_embd: TPTensor::from(patch_embd),
                patch_embd1: TPTensor::from(patch_embd1),
            }
        } else {
            todo!();
        }
    }
}

impl<T> NuralNetwork<T> for PatchEmbd<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        destruct!([x] = inputs);

        dims!([n, _c, height, width] = x);
        let Self {
            dt,
            shape,
            patch_embd,
            patch_embd1,
        } = self;
        let [m, ck, hk, wk] = shape.map(Dim::from);
        assert!(hk.eq(&wk));
        let w = ctx.load_external(
            "patch_embd",
            dt,
            [m.clone(), ck.clone(), hk.clone(), wk.clone()],
            patch_embd,
        );
        let w1 = ctx.load_external(
            "patch_embd1",
            dt,
            [m.clone(), ck.clone(), hk.clone(), wk.clone()],
            patch_embd1,
        );
        let tensors = ctx
            .call("", "conv", Some(false.into()), [x.clone(), w])
            .unwrap();
        destruct!([patch_embd] = tensors);
        let tensors = ctx.call("", "conv", Some(false.into()), [x, w1]).unwrap();
        destruct!([patch_embd1] = tensors);
        let tensors = ctx
            .call("", "add4d", None, [patch_embd, patch_embd1])
            .unwrap();
        destruct!([image_embd] = tensors);

        let hp = height.clone() / hk.clone(); // h patches
        let wp = width.clone() / wk.clone(); // w patches

        // transpose: [n, m, hp, wp] -> [n, hp, wp, m]
        destruct!([image_embd] = image_embd.transpose("", vec![0, 2, 3, 1]));

        // reshape: [n, hp, wp, m] -> [n * hp/2, 2, wp/2, 2*m]
        destruct!([image_embd] = image_embd.tile("", 1, [hp.clone() / 2, Dim::from(2)]));
        destruct!([image_embd] = image_embd.merge("", 0, 2));
        destruct!([image_embd] = image_embd.tile("", 2, [wp / 2, Dim::from(2)]));
        destruct!([image_embd] = image_embd.merge("", 3, 2));

        // transpose: [n * hp/2, 2, wp/2, 2*m] -> [n * hp/2, wp/2, 2, 2*m]
        destruct!([image_embd] = image_embd.transpose("", vec![0, 2, 1, 3]));

        // reshape: [n * hp/2, wp/2, 2, 2*m] -> [n, hp * wp, m]
        destruct!([image_embd] = image_embd.tile("", 0, [n.clone(), hp / 2]));
        destruct!([image_embd] = image_embd.merge("", 1, 3));
        destruct!([image_embd] = image_embd.tile("", 2, [Dim::from(2), m]));
        destruct!([image_embd] = image_embd.merge("", 1, 2));

        // merge-last: [n, hp * wp, m] -> [n * patches, m]
        destruct!([image_embd] = image_embd.merge("", 0, 2));

        Ok((ctx, vec![image_embd]))
    }
}
