use crate::{Args, NuralNetwork, Operator, Pc, TensorMeta, VirtualMachine};
use std::{collections::HashMap, marker::PhantomData};

pub struct Domain<'ctx, VM: VirtualMachine> {
    pid: u64,
    dev: u64,
    stack: Vec<Branch>,
    ops: HashMap<String, Box<dyn Operator<VM>>>,
    vm: &'ctx VM,
}

impl<'ctx, VM: VirtualMachine> Domain<'ctx, VM> {
    pub(crate) fn new<I>(pid: u64, dev: u64, ops: I, vm: &'ctx VM) -> Self
    where
        I: IntoIterator<Item = Box<dyn Operator<VM>>>,
    {
        Self {
            pid,
            dev,
            stack: Vec::new(),
            ops: ops.into_iter().map(|op| (op.name(), op)).collect(),
            vm,
        }
    }

    pub fn init<NN: NuralNetwork<VM>>(&mut self, meta: &NN::Meta, data: NN::Data) -> NN {
        NN::init(meta, data, self.ctx())
    }

    pub fn forward<NN: NuralNetwork<VM>>(&mut self, nn: &NN, args: NN::Args) {
        nn.forward(args, self.ctx())
    }

    fn ctx<NN: NuralNetwork<VM>>(&mut self) -> Context<VM, NN> {
        Context {
            pid: self.pid,
            dev: self.dev,
            stack: &mut self.stack,
            ops: &self.ops,
            vm: self.vm,
            _nn: PhantomData,
        }
    }
}

impl<VM: VirtualMachine> Drop for Domain<'_, VM> {
    fn drop(&mut self) {
        self.vm.drop_domain(self.pid, self.dev)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Branch {
    pub child_id: usize,
    pub loop_idx: usize,
}

pub struct Context<'ctx, VM: VirtualMachine, NN: NuralNetwork<VM>> {
    pid: u64,
    dev: u64,
    stack: &'ctx mut Vec<Branch>,
    ops: &'ctx HashMap<String, Box<dyn Operator<VM>>>,
    vm: &'ctx VM,
    _nn: PhantomData<NN>,
}

impl<VM: VirtualMachine, NN: NuralNetwork<VM>> Context<'_, VM, NN> {
    pub(crate) fn trap<Sub: NuralNetwork<VM>>(
        &mut self,
        child_id: usize,
        loop_idx: Option<usize>,
        name: &'static str,
    ) -> Context<VM, Sub> {
        let &mut Self {
            pid, dev, ops, vm, ..
        } = self;

        self.stack.push(Branch {
            child_id,
            loop_idx: loop_idx.unwrap_or(usize::MAX),
        });
        vm.record_trap(self.pos(), name, Sub::NAME);
        Context {
            pid,
            dev,
            stack: &mut self.stack,
            ops,
            vm,
            _nn: PhantomData,
        }
    }

    pub fn call(&self, op: &str, tensors: &[VM::Tensor], args: Box<dyn Args>) {
        self.vm.record_call(self.pos(), op, tensors, &*args);
        self.ops.get(op).unwrap().launch(tensors, args)
    }

    pub fn tensor(&self, meta: Option<TensorMeta>) -> VM::Tensor {
        self.vm.new_tensor(meta)
    }

    pub fn save_data(&self, data: VM::Tensor) {
        self.vm.save_tensor(self.pos(), data)
    }

    pub fn load_data(&self) -> VM::Tensor {
        self.vm.load_tensor(self.pos())
    }

    fn pos(&self) -> Pc {
        Pc {
            pid: self.pid,
            dev: self.dev,
            path: self.stack,
        }
    }
}

impl<VM: VirtualMachine, NN: NuralNetwork<VM>> Drop for Context<'_, VM, NN> {
    fn drop(&mut self) {
        self.stack.pop();
    }
}
