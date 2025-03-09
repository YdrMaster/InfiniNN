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

// #[cfg(test)]
// mod test {
//     use super::*;
//     use crate::test_recorder::TestVirtualMachiner;

//     impl Rearrange for TestVirtualMachiner {
//         fn rearrange(&self, y: &mut Tensor<Self::Blob>, x: &Tensor<Self::Blob>) {
//             assert_eq!(y.tensor.dt, x.tensor.dt);
//             assert_eq!(y.tensor.layout.shape(), x.tensor.layout.shape());

//             self.launch(format!(
//                 "rearrange(mut %{}, %{})",
//                 y.ptr.address(),
//                 x.ptr.address(),
//             ))
//         }
//     }

//     impl MatMul for TestVirtualMachiner {
//         fn mat_mul(
//             &self,
//             c: &mut Tensor<Self::Blob>,
//             beta: f32,
//             a: &Tensor<Self::Blob>,
//             b: &Tensor<Self::Blob>,
//             alpha: f32,
//         ) {
//             assert!(a.tensor.dt == c.tensor.dt && b.tensor.dt == c.tensor.dt);
//             match *c.tensor.layout.shape() {
//                 [m, n] => {
//                     let &[ma, ka] = a.tensor.layout.shape() else {
//                         panic!()
//                     };
//                     let &[kb, nb] = b.tensor.layout.shape() else {
//                         panic!()
//                     };
//                     assert_eq!(ma, m);
//                     assert_eq!(nb, n);
//                     assert_eq!(ka, kb)
//                 }
//                 [batch, m, n] => {
//                     let &[batch_a, ma, ka] = a.tensor.layout.shape() else {
//                         panic!()
//                     };
//                     let &[batch_b, kb, nb] = b.tensor.layout.shape() else {
//                         panic!()
//                     };
//                     assert_eq!(batch_a, batch);
//                     assert_eq!(batch_b, batch);
//                     assert_eq!(ma, m);
//                     assert_eq!(nb, n);
//                     assert_eq!(ka, kb)
//                 }
//                 [..] => panic!(),
//             }

//             self.launch(format!(
//                 "mat-mul(mut %{}, {beta:.2e}, %{}, %{}, {alpha:.2e})",
//                 c.ptr.address(),
//                 a.ptr.address(),
//                 b.ptr.address(),
//             ))
//         }
//     }

//     impl Softmax for TestVirtualMachiner {
//         fn softmax(&self, att: &mut Tensor<Self::Blob>, mask: AttnMask) {
//             assert_eq!(att.tensor.layout.ndim(), 3);

//             let mask = match mask {
//                 AttnMask::None => "",
//                 AttnMask::Causal => ", causal",
//             };
//             self.launch(format!("softmax(mut %{}{mask})", att.ptr.address()))
//         }
//     }

//     impl SwiGLU for TestVirtualMachiner {
//         fn swiglu(&self, gate: &mut Tensor<Self::Blob>, up: &Tensor<Self::Blob>) {
//             assert_eq!(gate.tensor.dt, up.tensor.dt);
//             assert_eq!(gate.tensor.layout.shape(), up.tensor.layout.shape());
//             assert_eq!(gate.tensor.layout.ndim(), 2);

//             self.launch(format!(
//                 "swiglu(mut %{}, %{})",
//                 gate.ptr.address(),
//                 up.ptr.address()
//             ))
//         }
//     }

//     impl GeLU for TestVirtualMachiner {
//         fn gelu(&self, up: &mut Tensor<Self::Blob>) {
//             assert_eq!(up.tensor.layout.ndim(), 2);

//             self.launch(format!("gelu(mut %{})", up.ptr.address()))
//         }
//     }

//     impl RmsNorm for TestVirtualMachiner {
//         fn rms_norm(
//             &self,
//             y: &mut Tensor<Self::Blob>,
//             x: &Tensor<Self::Blob>,
//             w: &Tensor<Self::Blob>,
//             epsilon: f32,
//         ) {
//             assert_eq!(y.tensor.dt, x.tensor.dt);
//             assert_eq!(y.tensor.layout.shape(), x.tensor.layout.shape());

//             self.launch(format!(
//                 "rms_norm(mut %{}, %{}, %{}, {epsilon:.2e})",
//                 y.ptr.address(),
//                 x.ptr.address(),
//                 w.ptr.address(),
//             ))
//         }
//     }

//     impl LayerNorm for TestVirtualMachiner {
//         fn layer_norm(
//             &self,
//             y: &mut Tensor<Self::Blob>,
//             x: &Tensor<Self::Blob>,
//             w: &Tensor<Self::Blob>,
//             b: &Tensor<Self::Blob>,
//         ) {
//             assert_eq!(y.tensor.dt, x.tensor.dt);
//             assert_eq!(y.tensor.layout.shape(), x.tensor.layout.shape());
//             assert_eq!(w.tensor.dt, b.tensor.dt);

//             self.launch(format!(
//                 "layer_norm(mut %{}, %{}, %{}, %{})",
//                 y.ptr.address(),
//                 x.ptr.address(),
//                 w.ptr.address(),
//                 b.ptr.address(),
//             ))
//         }
//     }

//     impl Add for TestVirtualMachiner {
//         fn add(&self, y: &mut Tensor<Self::Blob>, x: &Tensor<Self::Blob>) {
//             assert_eq!(y.tensor.dt, x.tensor.dt);
//             assert_eq!(y.tensor.layout.shape(), x.tensor.layout.shape());

//             self.launch(format!(
//                 "add(mut %{}, %{})",
//                 y.ptr.address(),
//                 x.ptr.address(),
//             ))
//         }
//     }
// }
