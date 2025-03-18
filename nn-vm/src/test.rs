use crate::{Args, Pc, Tensor, TensorMeta, VirtualMachine};

pub struct TestVM;
pub struct TestTensor;

impl Clone for TestTensor {
    fn clone(&self) -> Self {
        Self {}
    }
}

impl Tensor for TestTensor {
    fn assign(&self, other: Self) {
        todo!()
    }
}

impl VirtualMachine for TestVM {
    type Tensor = TestTensor;

    fn register(&self) -> u64 {
        todo!()
    }

    fn new_domain(&self, pid: u64, dev: u64) {
        todo!()
    }

    fn drop_domain(&self, pid: u64, dev: u64) {
        todo!()
    }

    fn record_trap(&self, pos: Pc, child_name: &str, nn_name: &str) {
        todo!()
    }

    fn record_call(&self, pos: Pc, op: &str, tensors: &[Self::Tensor], args: &dyn Args) {
        todo!()
    }

    fn new_tensor(&self, meta: Option<TensorMeta>) -> Self::Tensor {
        todo!()
    }

    fn save_tensor(&self, pos: Pc, tensor: Self::Tensor) {
        todo!()
    }

    fn load_tensor(&self, pos: Pc) -> Self::Tensor {
        todo!()
    }

    fn drop_tensor(&self, tensor: Self::Tensor) {
        todo!()
    }
}
