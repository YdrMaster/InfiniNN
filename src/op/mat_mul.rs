use crate::{Context, Tensor, VirtualMachine};

pub trait MatMul: VirtualMachine {
    fn mat_mul(
        &self,
        c: &mut Tensor<Self>,
        beta: f32,
        a: &Tensor<Self>,
        b: &Tensor<Self>,
        alpha: f32,
    );
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: MatMul + ?Sized,
{
    pub fn mat_mul(
        &self,
        c: &mut Tensor<VM>,
        beta: f32,
        a: &Tensor<VM>,
        b: &Tensor<VM>,
        alpha: f32,
    ) {
        self.vm.mat_mul(c, beta, a, b, alpha)
    }
}

#[cfg(test)]
impl MatMul for crate::test::TestVM {
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

        self.launch(format!(
            "mat-mul(mut %{}, {beta:.2e}, %{}, %{}, {alpha:.2e})",
            c.blob().id(),
            a.blob().id(),
            b.blob().id(),
        ))
    }
}
