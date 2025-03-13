use crate::{
    Context, Mapping, NuralNetwork, WeightBiasData,
    normalization::{self, Normalization},
};
use digit_layout::DigitLayout;
use std::ops::Deref;
use vm::{
    Tensor, VirtualMachine,
    op::{MatMul, Rearrange},
};

pub struct LmOutput {
    pub norm: Normalization,
    pub dt_w: DigitLayout,
}

pub struct Args<'vm, VM>
where
    VM: VirtualMachine + ?Sized,
{
    pub logit: Tensor<'vm, VM>,
    pub x: Tensor<'vm, VM>,
    pub requests: Vec<Request>,
}

pub struct Request {
    pub n_seq: usize,
    pub n_out: usize,
}

pub struct Data {
    pub norm: WeightBiasData,
    pub head: Box<dyn Deref<Target = [u8]>>,
}

pub trait Ops: Rearrange + normalization::Ops + MatMul {}
impl<VM> Ops for VM where VM: Rearrange + normalization::Ops + MatMul + ?Sized {}

impl<VM> NuralNetwork<VM> for LmOutput
where
    VM: VirtualMachine + ?Sized + Ops,
{
    type Args<'vm>
        = Args<'vm, VM>
    where
        VM: 'vm;
    type Data = Data;
    type Obj = ();
    type Sub = ();

    fn init(data: Self::Data, mut mapping: Mapping<VM, Self>) {
        let Self::Data { norm, head } = data;
        mapping.trap::<Normalization>((), norm).map_host((), head);
    }

    fn forward(&self, args: Self::Args<'_>, mut ctx: Context<VM, Self>) {
        let &Self { ref norm, dt_w } = self;
        let Args {
            mut logit,
            x,
            requests,
        } = args;

        // 集中要采样的 token
        let mut dst = 0;
        let mut src = 0;
        for req in &requests {
            src += req.n_seq - req.n_out;
            if src == dst {
                src += req.n_out;
                dst += req.n_out;
                continue;
            }
            let tgt = src + req.n_out;
            while src != tgt {
                let len = (src - dst).min(req.n_out);
                let mut dst_ = x.clone().slice(0, dst, len);
                let src_ = x.clone().slice(0, src, len);
                dst += len;
                src += len;
                ctx.rearrange(&mut dst_, &src_)
            }
        }

        let x = x.slice(0, 0, dst);
        let &[n, d] = x.shape() else { panic!() };
        let &[n_, dtok] = logit.shape() else { panic!() };
        assert_eq!(n, n_);

        ctx.trap(
            (),
            norm,
            normalization::Args {
                y: x.clone(),
                x: x.clone(),
            },
        );

        let w = ctx.get_mapped((), dt_w, &[dtok, d]).transpose(&[1, 0]);
        ctx.mat_mul(&mut logit, 0., &x, &w, 1.)
    }
}

#[cfg(test)]
mod test {
    use super::{Args, Data, LmOutput, Request};
    use crate::{
        VirtualMachineExt, WeightBiasData,
        normalization::{Normalization, Type},
    };
    use digit_layout::{DigitLayout, types};
    use test_vm::{TestVM, test_data};
    use vm::{VirtualMachine, device_id};

    const DEVICE: device_id = 0;
    const DT: DigitLayout = types::F16;
    const DT_NORM: DigitLayout = types::F32;
    const D: usize = 1024;
    const DTOK: usize = 32000;

    #[test]
    fn test() {
        let vm = TestVM::default();
        let pid = vm.register("lm-output");

        vm.init::<LmOutput>(
            pid,
            DEVICE,
            Data {
                norm: WeightBiasData {
                    weight: test_data(DT_NORM, &[D]),
                    bias: None,
                },
                head: test_data(DT, &[DTOK, D]),
            },
        )
        .forward(
            pid,
            DEVICE,
            &LmOutput {
                norm: Normalization {
                    ty: Type::RmsNorm { epsilon: 1e-5 },
                    dt_w: DT_NORM,
                },
                dt_w: DT,
            },
            Args {
                logit: vm.workspace(DT, &[5, DTOK]),
                x: vm.workspace(DT, &[19, D]),
                requests: vec![
                    Request { n_seq: 7, n_out: 1 },
                    Request { n_seq: 1, n_out: 1 },
                    Request {
                        n_seq: 11,
                        n_out: 3,
                    },
                ],
            },
        );

        vm.unregister(pid)
    }
}
