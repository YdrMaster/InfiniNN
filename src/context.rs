use crate::{Id, Tensor, VirtualMachine, nn::NuralNetwork, pid};
use digit_layout::DigitLayout;
use std::marker::PhantomData;

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

    pub fn workspace(&self, which: NN::Obj, dt: DigitLayout, shape: &[usize]) -> Tensor<'vm, VM> {
        let size = shape.iter().product::<usize>() * dt.nbytes() / dt.group_size();
        let blob = self.vm.alloc(self.stack.obj_id(which), size);
        Tensor::new(dt, shape, blob, self.vm)
    }

    pub fn get_mapped(&self, which: NN::Obj, dt: DigitLayout, shape: &[usize]) -> Tensor<'vm, VM> {
        let blob = self.vm.get_mapped(self.stack.obj_id(which));
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
}

impl<VM: VirtualMachine> Exec for VM {}

#[derive(Clone)]
pub struct ObjId(Box<[u8]>);

impl ObjId {
    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

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

    fn obj_id(&self, which: impl Id) -> ObjId {
        let mut stack = self.0.clone();
        let slice = which.as_slice();
        stack.push(slice.len() as _);
        stack.extend_from_slice(slice);
        ObjId(stack.into_boxed_slice())
    }
}
