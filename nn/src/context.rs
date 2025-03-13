use crate::NuralNetwork;
use digit_layout::DigitLayout;
use std::{marker::PhantomData, ops::Deref};
use vm::{ObjId, StackTracer, Tensor, VirtualMachine, device_id, pid};

#[repr(transparent)]
pub struct Mapping<'vm, VM: ?Sized, NN>(State<'vm, VM, NN>);

#[repr(transparent)]
pub struct Context<'vm, VM: ?Sized, NN>(State<'vm, VM, NN>);

struct State<'vm, VM: ?Sized, NN> {
    stack: &'vm mut StackTracer,
    vm: &'vm VM,
    _nn: PhantomData<NN>,
}

pub trait VirtualMachineExt: VirtualMachine {
    fn init<NN: NuralNetwork<Self>>(&self, pid: pid, dev: device_id, data: NN::Data) -> &Self {
        let mut stack = StackTracer::new(pid, dev);

        NN::init(
            data,
            Mapping(State {
                stack: &mut stack,
                vm: self,
                _nn: PhantomData,
            }),
        );

        self
    }

    fn forward<NN: NuralNetwork<Self>>(
        &self,
        pid: u64,
        dev: device_id,
        nn: &NN,
        args: NN::Args<'_>,
    ) -> &Self {
        let mut stack = StackTracer::new(pid, dev);

        nn.forward(
            args,
            Context(State {
                stack: &mut stack,
                vm: self,
                _nn: PhantomData,
            }),
        );

        self
    }

    fn workspace<'vm>(&'vm self, dt: DigitLayout, shape: &[usize]) -> Tensor<'vm, Self> {
        let size = shape.iter().product::<usize>() * dt.nbytes() / dt.group_size();
        let blob = self.alloc(ObjId::global(), size);
        Tensor::new(dt, shape, blob, self)
    }
}

impl<VM: VirtualMachine> VirtualMachineExt for VM {}

impl<VM: ?Sized, NN> Context<'_, VM, NN> {
    pub fn stack(&self) -> ObjId {
        self.0.stack.path()
    }

    pub fn vm(&self) -> &VM {
        self.0.vm
    }
}

impl<'vm, VM, NN> Context<'vm, VM, NN>
where
    VM: VirtualMachine + ?Sized,
    NN: NuralNetwork<VM>,
{
    pub fn trap<Sub: NuralNetwork<VM>>(
        &mut self,
        id: NN::Sub,
        sub: &Sub,
        args: Sub::Args<'_>,
    ) -> &mut Self {
        self.0.stack.push(id);

        sub.forward(
            args,
            Context(State {
                stack: self.0.stack,
                vm: self.0.vm,
                _nn: PhantomData,
            }),
        );

        self.0.stack.pop();
        self
    }

    pub fn workspace(&self, dt: DigitLayout, shape: &[usize]) -> Tensor<'vm, VM> {
        let size = shape.iter().product::<usize>() * dt.nbytes() / dt.group_size();
        let blob = self.0.vm.alloc(self.0.stack.path(), size);
        Tensor::new(dt, shape, blob, self.0.vm)
    }

    pub fn map_host(
        &self,
        dt: DigitLayout,
        shape: &[usize],
        data: Box<dyn Deref<Target = [u8]>>,
    ) -> Tensor<'vm, VM> {
        let blob = self.0.vm.map_host(self.0.stack.path(), data);
        Tensor::new(dt, shape, blob, self.0.vm)
    }

    pub fn get_mapped(&self, which: NN::Obj, dt: DigitLayout, shape: &[usize]) -> Tensor<'vm, VM> {
        let blob = self.0.vm.get_mapped(self.0.stack.obj(which));
        Tensor::new(dt, shape, blob, self.0.vm)
    }
}

impl<VM, NN> Mapping<'_, VM, NN>
where
    VM: VirtualMachine + ?Sized,
    NN: NuralNetwork<VM>,
{
    pub fn trap<Sub: NuralNetwork<VM>>(&mut self, id: NN::Sub, data: Sub::Data) -> &mut Self {
        self.0.stack.push(id);

        Sub::init(
            data,
            Mapping(State {
                stack: self.0.stack,
                vm: self.0.vm,
                _nn: PhantomData,
            }),
        );

        self.0.stack.pop();
        self
    }

    pub fn map_host(&mut self, which: NN::Obj, mem: Box<dyn Deref<Target = [u8]>>) -> &mut Self {
        self.0
            .vm
            .free(self.0.vm.map_host(self.0.stack.obj(which), mem));
        self
    }
}
