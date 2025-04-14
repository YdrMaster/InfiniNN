pub mod embedding;

use crate::{NNTensor, VirtualMachine};
use std::any::Any;

pub trait Operator<VM: VirtualMachine> {
    type Args: Any;

    fn call(
        args: &Self::Args,
        inputs: impl IntoIterator<Item = NNTensor<VM>>,
        domain: &VM::Domain,
    ) -> Vec<NNTensor<VM>>;
}
