use crate::{Id, Tensor, VirtualMachine, dev_id, nn::NuralNetwork, pid};
use digit_layout::DigitLayout;
use std::{fmt, marker::PhantomData, ops::Deref};

pub struct Context<'vm, VM: ?Sized, NN> {
    stack: &'vm mut Vec<u8>,
    pub(crate) vm: &'vm VM,
    _nn: PhantomData<NN>,
}

impl<VM: ?Sized, NN> Context<'_, VM, NN> {
    pub fn stack(&self) -> ObjId {
        stack(self.stack)
    }

    fn obj_id(&self, which: impl Id) -> ObjId {
        obj_id(self.stack, which)
    }
}

impl<'vm, VM, NN> Context<'vm, VM, NN>
where
    VM: VirtualMachine + ?Sized,
    NN: NuralNetwork<VM>,
{
    pub fn trap<Sub: NuralNetwork<VM>>(&mut self, id: NN::Sub, sub: &Sub, args: Sub::Args<'_>) {
        push(self.stack, id);

        sub.launch(
            args,
            Context {
                stack: self.stack,
                vm: self.vm,
                _nn: PhantomData,
            },
        );

        pop::<NN::Sub>(self.stack)
    }

    pub fn workspace(&self, dt: DigitLayout, shape: &[usize]) -> Tensor<'vm, VM> {
        let size = shape.iter().product::<usize>() * dt.nbytes() / dt.group_size();
        let blob = self.vm.alloc(self.stack(), size);
        Tensor::new(dt, shape, blob, self.vm)
    }

    pub fn get_mapped(&self, which: NN::Obj, dt: DigitLayout, shape: &[usize]) -> Tensor<'vm, VM> {
        let blob = self.vm.get_mapped(self.obj_id(which));
        Tensor::new(dt, shape, blob, self.vm)
    }
}

pub trait Exec: VirtualMachine {
    /// 在指定设备上执行一个网络。
    fn exec<NN: NuralNetwork<Self>>(&self, pid: u64, device: dev_id, nn: &NN, args: NN::Args<'_>) {
        let mut stack = StackHeader { pid, device }.as_slice().to_vec();

        nn.launch(
            args,
            Context {
                stack: &mut stack,
                vm: self,
                _nn: PhantomData,
            },
        )
    }

    fn workspace<'vm>(
        &'vm self,
        device: Option<dev_id>,
        dt: DigitLayout,
        shape: &[usize],
    ) -> Tensor<'vm, Self> {
        let header = StackHeader {
            pid: pid::MAX,
            device: device.unwrap_or(dev_id::MAX),
        };
        let obj = stack(header.as_slice());
        let size = shape.iter().product::<usize>() * dt.nbytes() / dt.group_size();
        let blob = self.alloc(obj, size);
        Tensor::new(dt, shape, blob, self)
    }
}

impl<VM: VirtualMachine> Exec for VM {}

pub trait Map: VirtualMachine {
    /// 在指定设备上启动一个上下文。
    fn map<NN: NuralNetwork<Self>>(&self, pid: u64, device: dev_id) -> Mapping<Self, NN> {
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
        push(&mut stack, sub);
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
        self.0[self.0.len() - 1] == 0
    }

    pub fn domain(&self) -> String {
        unsafe { self.0.as_ptr().cast::<StackHeader>().read_unaligned() }.to_string()
    }

    pub fn body(&self) -> String {
        StackBody(&self.0[size_of::<StackHeader>()..]).to_string()
    }
}

fn push(vec: &mut Vec<u8>, id: impl Id) {
    let slice = id.as_slice();
    vec.push(slice.len() as _);
    vec.extend_from_slice(slice);
}

fn pop<T: Id>(vec: &mut Vec<u8>) {
    vec.truncate(vec.len() - size_of::<T>() - 1)
}

fn stack(stack: &[u8]) -> ObjId {
    let mut stack = stack.to_vec();
    stack.push(0);
    ObjId(stack.into())
}

fn obj_id(stack: &[u8], which: impl Id) -> ObjId {
    let mut stack = stack.to_vec();
    push(&mut stack, which);
    stack.push(1);
    ObjId(stack.into())
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct StackHeader {
    pid: pid,
    device: dev_id,
}

impl fmt::Display for StackHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let &Self { pid, device } = self;
        if pid == pid::MAX {
            write!(f, "vm:")?
        } else {
            write!(f, "#{pid}:")?
        }
        if device == dev_id::MAX {
            write!(f, "H")
        } else {
            write!(f, "{device}")
        }
    }
}

struct StackBody<'a>(&'a [u8]);

impl fmt::Display for StackBody<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut slice = self.0;
        let stack = slice[slice.len() - 1] == 0;
        slice = &slice[..slice.len() - 1];
        while let &[len, ref tail @ ..] = slice {
            write!(f, ".")?;
            if len == 0 {
                write!(f, "()")?;
                slice = tail
            } else {
                let (it, tail) = tail.split_at(len as _);
                for &byte in it {
                    write!(f, "{byte:02x}")?
                }
                slice = tail
            }
        }
        if stack {
            write!(f, ".*")?
        }
        Ok(())
    }
}
