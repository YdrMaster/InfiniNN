use crate::{Id, Tensor, VirtualMachine, nn::NuralNetwork, pid};
use digit_layout::DigitLayout;
use std::{marker::PhantomData, ops::Deref};

pub struct Context<'vm, VM: ?Sized, NN> {
    stack: Stack<'vm>,
    pub(crate) vm: &'vm VM,
    _nn: PhantomData<NN>,
}

impl<'vm, VM, NN> Context<'vm, VM, NN>
where
    VM: VirtualMachine + ?Sized,
    NN: NuralNetwork<VM>,
{
    pub fn trap<Sub: NuralNetwork<VM>>(&mut self, id: NN::Sub, sub: &Sub, args: Sub::Args<'_, '_>) {
        let stack = self.stack.push(id);

        sub.launch(
            args,
            Context {
                stack,
                vm: self.vm,
                _nn: PhantomData,
            },
        );

        self.stack.pop::<NN::Sub>()
    }

    pub fn workspace(&self, dt: DigitLayout, shape: &[usize]) -> Tensor<'vm, VM> {
        let size = shape.iter().product::<usize>() * dt.nbytes() / dt.group_size();
        let blob = self.vm.alloc(self.stack.free_obj(), size);
        Tensor::new(dt, shape, blob, self.vm)
    }

    pub fn get_mapped(&self, which: NN::Obj, dt: DigitLayout, shape: &[usize]) -> Tensor<'vm, VM> {
        let blob = self.vm.get_mapped(obj_id(self.stack.0, which));
        Tensor::new(dt, shape, blob, self.vm)
    }
}

pub trait Exec: VirtualMachine {
    /// 在指定设备上启动一个上下文。
    fn exec<NN: NuralNetwork<Self>>(
        &self,
        pid: u64,
        device: usize,
        nn: &NN,
        args: NN::Args<'_, '_>,
    ) {
        let mut stack = pid.as_slice().to_vec();
        stack.extend_from_slice(device.as_slice());

        nn.launch(
            args,
            Context {
                stack: Stack(&mut stack),
                vm: self,
                _nn: PhantomData,
            },
        )
    }

    fn workspace<'vm>(
        &'vm self,
        device: Option<usize>,
        dt: DigitLayout,
        shape: &[usize],
    ) -> Tensor<'vm, Self> {
        let obj = if let Some(device) = device {
            ObjId(device.as_slice().to_vec().into())
        } else {
            ObjId(Box::new([]))
        };
        let size = shape.iter().product::<usize>() * dt.nbytes() / dt.group_size();
        let blob = self.alloc(obj, size);
        Tensor::new(dt, shape, blob, self)
    }
}

impl<VM: VirtualMachine> Exec for VM {}

pub trait Map: VirtualMachine {
    /// 在指定设备上启动一个上下文。
    fn map<NN: NuralNetwork<Self>>(&self, pid: u64, device: usize) -> Mapping<Self, NN> {
        let mut stack = pid.as_slice().to_vec();
        stack.extend_from_slice(device.as_slice());
        Mapping {
            stack,
            _nn: PhantomData,
            vm: self,
        }
    }
}

impl<VM: VirtualMachine> Map for VM {}

pub struct Mapping<'vm, VM: ?Sized, NN> {
    stack: Vec<u8>,
    _nn: PhantomData<NN>,
    vm: &'vm VM,
}

impl<VM, NN> Mapping<'_, VM, NN>
where
    VM: VirtualMachine + ?Sized,
    NN: NuralNetwork<VM>,
{
    pub fn step_into<Sub: NuralNetwork<VM>>(&self, sub: NN::Sub) -> Mapping<VM, Sub> {
        let mut stack = self.stack.clone();
        Stack(&mut stack).push(sub);
        Mapping {
            stack,
            _nn: PhantomData,
            vm: self.vm,
        }
    }

    pub fn map_host(&self, which: NN::Obj, mem: Box<dyn Deref<Target = [u8]>>) {
        self.vm.map_host(obj_id(&self.stack, which), mem);
    }
}

#[derive(Clone)]
pub struct ObjId(Box<[u8]>);

impl ObjId {
    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

    pub fn is_free(&self) -> bool {
        self.0.len() <= size_of::<pid>() + size_of::<usize>()
    }

    pub fn device(&self) -> Option<usize> {
        if self.0.len() >= size_of::<usize>() {
            Some(usize::from_slice(&self.0[..size_of::<usize>()]))
        } else {
            None
        }
    }

    pub fn pid(&self) -> Option<pid> {
        if self.0.len() >= size_of::<usize>() + size_of::<pid>() {
            Some(pid::from_slice(
                &self.0[size_of::<usize>()..][..size_of::<pid>()],
            ))
        } else {
            None
        }
    }

    pub fn domain(&self) -> String {
        self.pid().map_or("vm".into(), |pid| format!("#{pid:x}"))
    }
}

struct Stack<'a>(&'a mut Vec<u8>);

impl Stack<'_> {
    fn push(&mut self, id: impl Id) -> Stack {
        let slice = id.as_slice();
        self.0.push(slice.len() as _);
        self.0.extend_from_slice(slice);
        Stack(self.0)
    }

    fn pop<T: Id>(&mut self) {
        self.0.truncate(self.0.len() - size_of::<T>())
    }

    fn free_obj(&self) -> ObjId {
        ObjId(
            self.0[..size_of::<pid>() + size_of::<usize>()]
                .to_vec()
                .into(),
        )
    }
}

fn obj_id(vec: &[u8], which: impl Id) -> ObjId {
    let mut stack = vec.to_vec();
    let slice = which.as_slice();
    stack.push(slice.len() as _);
    stack.extend_from_slice(slice);
    ObjId(stack.into_boxed_slice())
}
