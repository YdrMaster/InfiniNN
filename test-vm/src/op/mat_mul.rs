use vm::{ObjId, Tensor, op::MatMul};

impl MatMul for crate::TestVM {
    fn mat_mul(
        &self,
        stack: ObjId,
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

        self.launch(
            stack,
            format!(
                "mat-mul(mut %{}, {beta:.2e}, %{}, %{}, {alpha:.2e})",
                c.blob().id(),
                a.blob().id(),
                b.blob().id(),
            ),
        )
    }
}
