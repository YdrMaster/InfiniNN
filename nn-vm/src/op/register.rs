use super::{Args, Operator};
use crate::VirtualMachine;
use std::{collections::HashMap, sync::RwLock};

pub struct OpRegister<VM: VirtualMachine>(RwLock<HashMap<String, Box<dyn Operator<VM>>>>);

impl<VM> OpRegister<VM>
where
    VM: VirtualMachine,
{
    pub fn register(&mut self, op: Box<dyn Operator<VM>>) {
        let mut map = self.0.write().unwrap();
        let type_id = op.type_id();
        let previous = map.insert(op.name(), op);
        if let Some(previous) = previous {
            assert_eq!(previous.type_id(), type_id)
        }
    }

    pub fn call(&self, name: &str, tensors: &[&VM::Tensor], args: Box<dyn Args>) {
        let vec = self.0.read().unwrap();
        if let Some(op) = vec.get(name) {
            op.launch(tensors, args)
        } else {
            unreachable!("Op not found")
        }
    }
}
