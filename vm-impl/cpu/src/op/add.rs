use vm::{ObjId, Tensor, op::Add};

impl Add for crate::CpuVM {
    fn add(&self, _stack: ObjId, y: &mut Tensor<Self>, x: &Tensor<Self>) {
        assert_eq!(y.dt(), x.dt());
        assert_eq!(y.shape(), x.shape());

        todo!()
    }
}
