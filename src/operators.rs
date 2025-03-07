use crate::{MemManage, StorageTensor};

pub trait Rearrange: MemManage {
    fn rearrange(&self, y: &mut StorageTensor<Self::B>, x: &StorageTensor<Self::B>);
}

pub trait MatMul: MemManage {
    fn mat_mul(
        &self,
        c: &mut StorageTensor<Self::B>,
        beta: f32,
        a: &StorageTensor<Self::B>,
        b: &StorageTensor<Self::B>,
        alpha: f32,
    );
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum AttnMask {
    None,
    Causal,
}

pub trait Softmax: MemManage {
    fn softmax(&self, att: &mut StorageTensor<Self::B>, mask: AttnMask);
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_recorder::TestMemManager;

    impl Rearrange for TestMemManager {
        fn rearrange(&self, y: &mut StorageTensor<Self::B>, x: &StorageTensor<Self::B>) {
            assert_eq!(y.tensor.dt, x.tensor.dt);
            assert_eq!(y.tensor.layout.shape(), x.tensor.layout.shape());
            self.launch(format!(
                "rearrange(mut %{}, %{})",
                y.ptr.address(),
                x.ptr.address(),
            ))
        }
    }

    impl MatMul for TestMemManager {
        fn mat_mul(
            &self,
            c: &mut StorageTensor<Self::B>,
            beta: f32,
            a: &StorageTensor<Self::B>,
            b: &StorageTensor<Self::B>,
            alpha: f32,
        ) {
            assert!(a.tensor.dt == c.tensor.dt && b.tensor.dt == c.tensor.dt);
            match *c.tensor.layout.shape() {
                [m, n] => {
                    let &[ma, ka] = a.tensor.layout.shape() else {
                        panic!()
                    };
                    let &[kb, nb] = b.tensor.layout.shape() else {
                        panic!()
                    };
                    assert_eq!(ma, m);
                    assert_eq!(nb, n);
                    assert_eq!(ka, kb)
                }
                [batch, m, n] => {
                    let &[batch_a, ma, ka] = a.tensor.layout.shape() else {
                        panic!()
                    };
                    let &[batch_b, kb, nb] = b.tensor.layout.shape() else {
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
                c.ptr.address(),
                a.ptr.address(),
                b.ptr.address(),
            ))
        }
    }

    impl Softmax for TestMemManager {
        fn softmax(&self, att: &mut StorageTensor<Self::B>, mask: AttnMask) {
            assert_eq!(att.tensor.layout.ndim(), 3);
            let mask = match mask {
                AttnMask::None => "",
                AttnMask::Causal => ", causal",
            };
            self.launch(format!("softmax(mut %{}{mask})", att.ptr.address()))
        }
    }
}
