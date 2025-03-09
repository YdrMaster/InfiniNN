mod add;
mod gelu;
mod layer_norm;
mod mat_mul;
mod rearrange;
mod rms_norm;
mod softmax;
mod swiglu;

pub use add::Add;
pub use gelu::GeLU;
pub use layer_norm::LayerNorm;
pub use mat_mul::MatMul;
pub use rearrange::Rearrange;
pub use rms_norm::RmsNorm;
pub use softmax::{AttnMask, Softmax};
pub use swiglu::SwiGLU;

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Tensor, test::TestVM};

    impl Rearrange for TestVM {
        fn rearrange(&self, y: &mut Tensor<Self>, x: &Tensor<Self>) {
            assert_eq!(y.dt(), x.dt());
            assert_eq!(y.shape(), x.shape());

            // self.launch(format!(
            //     "rearrange(mut %{}, %{})",
            //     y.ptr.address(),
            //     x.ptr.address(),
            // ))
        }
    }

    impl MatMul for TestVM {
        fn mat_mul(
            &self,
            c: &mut Tensor<Self>,
            beta: f32,
            a: &Tensor<Self>,
            b: &Tensor<Self>,
            alpha: f32,
        ) {
            assert!(a.dt() == c.dt() && b.dt() == c.dt());
            match *c.shape() {
                [m, n] => {
                    let &[ma, ka] = a.shape() else { panic!() };
                    let &[kb, nb] = b.shape() else { panic!() };
                    assert_eq!(ma, m);
                    assert_eq!(nb, n);
                    assert_eq!(ka, kb)
                }
                [batch, m, n] => {
                    let &[batch_a, ma, ka] = a.shape() else {
                        panic!()
                    };
                    let &[batch_b, kb, nb] = b.shape() else {
                        panic!()
                    };
                    assert_eq!(batch_a, batch);
                    assert_eq!(batch_b, batch);
                    assert_eq!(ma, m);
                    assert_eq!(nb, n);
                    assert_eq!(ka, kb)
                }
                [..] => panic!(),
            }

            // self.launch(format!(
            //     "mat-mul(mut %{}, {beta:.2e}, %{}, %{}, {alpha:.2e})",
            //     c.ptr.address(),
            //     a.ptr.address(),
            //     b.ptr.address(),
            // ))
        }
    }

    impl Softmax for TestVM {
        fn softmax(&self, att: &mut Tensor<Self>, mask: AttnMask) {
            assert_eq!(att.shape().len(), 3);

            let _mask = match mask {
                AttnMask::None => "",
                AttnMask::Causal => ", causal",
            };
            // self.launch(format!("softmax(mut %{}{mask})", att.ptr.address()))
        }
    }

    impl SwiGLU for TestVM {
        fn swiglu(&self, gate: &mut Tensor<Self>, up: &Tensor<Self>) {
            assert_eq!(gate.dt(), up.dt());
            assert_eq!(gate.shape(), up.shape());
            assert_eq!(gate.shape().len(), 2);

            // self.launch(format!(
            //     "swiglu(mut %{}, %{})",
            //     gate.ptr.address(),
            //     up.ptr.address()
            // ))
        }
    }

    impl GeLU for TestVM {
        fn gelu(&self, up: &mut Tensor<Self>) {
            assert_eq!(up.shape().len(), 2);

            // self.launch(format!("gelu(mut %{})", up.ptr.address()))
        }
    }

    impl RmsNorm for TestVM {
        fn rms_norm(&self, y: &mut Tensor<Self>, x: &Tensor<Self>, w: &Tensor<Self>, epsilon: f32) {
            assert_eq!(y.dt(), x.dt());
            assert_eq!(y.shape(), x.shape());

            // self.launch(format!(
            //     "rms_norm(mut %{}, %{}, %{}, {epsilon:.2e})",
            //     y.ptr.address(),
            //     x.ptr.address(),
            //     w.ptr.address(),
            // ))
        }
    }

    impl LayerNorm for TestVM {
        fn layer_norm(
            &self,
            y: &mut Tensor<Self>,
            x: &Tensor<Self>,
            w: &Tensor<Self>,
            b: &Tensor<Self>,
        ) {
            assert_eq!(y.dt(), x.dt());
            assert_eq!(y.shape(), x.shape());
            assert_eq!(w.dt(), b.dt());

            // self.launch(format!(
            //     "layer_norm(mut %{}, %{}, %{}, %{})",
            //     y.ptr.address(),
            //     x.ptr.address(),
            //     w.ptr.address(),
            //     b.ptr.address(),
            // ))
        }
    }

    impl Add for TestVM {
        fn add(&self, y: &mut Tensor<Self>, x: &Tensor<Self>) {
            assert_eq!(y.dt(), x.dt());
            assert_eq!(y.shape(), x.shape());

            // self.launch(format!(
            //     "add(mut %{}, %{})",
            //     y.ptr.address(),
            //     x.ptr.address(),
            // ))
        }
    }
}
