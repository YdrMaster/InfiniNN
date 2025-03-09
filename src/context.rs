use crate::{Id, VirtualMachine, nn::NuralNetwork, pid};
use std::marker::PhantomData;

pub struct Context<'vm, VM: ?Sized, NN> {
    stack: Stack<'vm>,
    _nn: PhantomData<NN>,
    pub(super) vm: &'vm VM,
}

impl<VM, NN> Context<'_, VM, NN>
where
    VM: VirtualMachine + ?Sized,
    NN: NuralNetwork<VM>,
{
    pub fn trap<Sub: NuralNetwork<VM>>(&mut self, sub: NN::Sub, args: Sub::Args) {
        let stack = self.stack.push(sub);
        let ctx = Context {
            stack,
            _nn: PhantomData,
            vm: self.vm,
        };
        Sub::launch(ctx, args);
        self.stack.pop::<NN::Sub>()
    }
}

pub trait Exec: VirtualMachine {
    /// 在指定设备上启动一个上下文。
    fn exec<NN: NuralNetwork<Self>>(&self, pid: u64, device: usize, args: NN::Args) {
        let mut stack = pid.as_slice().to_vec();
        stack.extend_from_slice(device.as_slice());
        let ctx = Context {
            stack: Stack(&mut stack),
            _nn: PhantomData,
            vm: self,
        };
        NN::launch(ctx, args)
    }
}

impl<VM: VirtualMachine> Exec for VM {}

pub struct ObjId(Box<[u8]>);

impl ObjId {
    pub fn pid(&self) -> pid {
        pid::from_slice(&self.0[..size_of::<pid>()])
    }

    pub fn device(&self) -> usize {
        usize::from_slice(&self.0[size_of::<pid>()..][..size_of::<usize>()])
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
}
