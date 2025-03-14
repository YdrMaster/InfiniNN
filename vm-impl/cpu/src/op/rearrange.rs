use mem_rearrange::Rearranging;
use vm::{ObjId, Tensor, op::Rearrange};

impl Rearrange for crate::CpuVM {
    fn rearrange(&self, _stack: ObjId, y: &mut Tensor<Self>, x: &Tensor<Self>) {
        let dt = Tensor::check_dt_same(&[y, x]).unwrap();
        unsafe {
            Rearranging::new(y.layout(), x.layout(), dt.nbytes())
                .unwrap()
                .launch(y.blob().as_ptr().cast_mut(), x.blob().as_ptr())
        }
    }
}
