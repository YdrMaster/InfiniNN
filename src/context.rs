use crate::{Id, Tensor, VirtualMachine, dev_id, nn::NuralNetwork, pid};
use digit_layout::DigitLayout;
use std::{fmt, marker::PhantomData, ops::Deref};

pub struct Context<'vm, VM: ?Sized, NN> {
    stack: &'vm mut String,
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
        let len = push(self.stack, id);

        sub.launch(
            args,
            Context {
                stack: self.stack,
                vm: self.vm,
                _nn: PhantomData,
            },
        );

        pop(self.stack, len)
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
    fn exec<NN: NuralNetwork<Self>>(&self, pid: u64, dev_id: dev_id, nn: &NN, args: NN::Args<'_>) {
        let mut stack = Domain { pid, dev_id }.to_string();

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
        let header = Domain {
            pid: pid::MAX,
            dev_id: device.unwrap_or(dev_id::MAX),
        }
        .to_string();
        let obj = stack(&header);
        let size = shape.iter().product::<usize>() * dt.nbytes() / dt.group_size();
        let blob = self.alloc(obj, size);
        Tensor::new(dt, shape, blob, self)
    }
}

impl<VM: VirtualMachine> Exec for VM {}

pub trait Map: VirtualMachine {
    /// 在指定设备上启动一个上下文。
    fn map<NN: NuralNetwork<Self>>(&self, pid: pid, dev_id: dev_id) -> Mapping<Self, NN> {
        Mapping {
            stack: Domain { pid, dev_id }.to_string(),
            _nn: PhantomData,
            vm: self,
        }
    }
}

impl<VM: VirtualMachine> Map for VM {}

pub struct Mapping<'vm, VM: ?Sized, NN> {
    stack: String,
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
        self.vm
            .free(self.vm.map_host(obj_id(&self.stack, which), mem))
    }
}

#[derive(Clone)]
pub struct ObjId {
    text: String,
    _is_obj: bool,
}

impl ObjId {
    pub fn as_str(&self) -> &str {
        &self.text
    }

    pub fn domain(&self) -> &str {
        let idx = self.text.find(']').unwrap() + 1;
        &self.text[..idx]
    }

    pub fn body(&self) -> &str {
        let idx = self.text.find(']').unwrap() + 1;
        &self.text[idx..]
    }
}

fn push(vec: &mut String, id: impl Id) -> usize {
    let len = vec.len();
    vec.push('.');
    vec.push_str(id.name());
    if let Some(idx) = id.idx() {
        vec.push('#');
        vec.push_str(&idx.to_string())
    }
    len
}

fn pop(vec: &mut String, len: usize) {
    vec.truncate(len)
}

fn obj_id(stack: &str, id: impl Id) -> ObjId {
    let mut text = stack.to_string();
    push(&mut text, id);
    ObjId {
        text,
        _is_obj: true,
    }
}

fn stack(stack: &str) -> ObjId {
    ObjId {
        text: stack.to_string(),
        _is_obj: false,
    }
}

struct Domain {
    pid: pid,
    dev_id: dev_id,
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let &Self { pid, dev_id } = self;
        write!(f, "[")?;
        match pid {
            pid::MAX => write!(f, "vm")?,
            n => write!(f, "#{n:x}")?,
        }
        write!(f, ":")?;
        match dev_id {
            dev_id::MAX => write!(f, "H")?,
            n => write!(f, "{n}")?,
        }
        write!(f, "].")
    }
}
