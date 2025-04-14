use super::Operator;
use crate::{Domain, NNTensor, VirtualMachine, macros::dims};
use tensor::digit_layout::LayoutContent;

pub struct Forward;
pub struct Backward;

impl<VM: VirtualMachine> Operator<VM> for Forward {
    type Args = ();

    fn call(
        (): &Self::Args,
        inputs: impl IntoIterator<Item = NNTensor<VM>>,
        domain: &VM::Domain,
    ) -> Vec<NNTensor<VM>> {
        let mut inputs = inputs.into_iter();
        let wte = inputs.next().unwrap();
        let tokens = inputs.next().unwrap();

        assert!(matches!(
            tokens.dt().decode(),
            LayoutContent::Unsigned { .. }
        ));

        dims!([_, d] = wte);
        dims!([n] = tokens);

        let mut ans = domain.tensor(wte.dt(), &[n, d]);

        match inputs.next() {
            Some(wpe) => {
                let pos = inputs.next().unwrap();
                domain.launch(
                    "embedding",
                    [
                        ans.kernel_mut(),
                        wte.kernel_ref(),
                        tokens.kernel_ref(),
                        wpe.kernel_ref(),
                        pos.kernel_ref(),
                    ],
                    [],
                )
            }
            None => domain.launch(
                "embedding",
                [ans.kernel_mut(), wte.kernel_ref(), tokens.kernel_ref()],
                [],
            ),
        }

        vec![ans.into()]
    }
}

impl<VM: VirtualMachine> Operator<VM> for Backward {
    type Args = ();

    fn call(
        (): &Self::Args,
        inputs: impl IntoIterator<Item = NNTensor<VM>>,
        domain: &VM::Domain,
    ) -> Vec<NNTensor<VM>> {
        let mut inputs = inputs.into_iter();
        let t0 = inputs.next();
        let t1 = inputs.next();
        let t2 = inputs.next();
        let t3 = inputs.next();
        let t4 = inputs.next();

        if t3.is_some() {
            let mut dwte = t0.unwrap();
            let mut dwpe = t1.unwrap();
            let dy = t2.unwrap();
            let tokens = t3.unwrap();
            let pos = t4.unwrap();

            domain.launch(
                "embedding-backward",
                [
                    dwte.kernel_mut(),
                    dwpe.kernel_mut(),
                    dy.kernel_ref(),
                    tokens.kernel_ref(),
                    pos.kernel_ref(),
                ],
                [],
            )
        } else {
            let mut dwte = t0.unwrap();
            let dy = t2.unwrap();
            let tokens = t3.unwrap();

            domain.launch(
                "embedding-backward",
                [dwte.kernel_mut(), dy.kernel_ref(), tokens.kernel_ref()],
                [],
            )
        }

        vec![]
    }
}
