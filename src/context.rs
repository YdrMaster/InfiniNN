use crate::{Id, Tensor, VirtualMachine, dev_id, nn::NuralNetwork, pid};
use digit_layout::DigitLayout;
use std::{fmt, marker::PhantomData, ops::Deref};

#[repr(transparent)]
pub struct Mapping<'vm, VM: ?Sized, NN>(State<'vm, VM, NN>);

#[repr(transparent)]
pub struct Context<'vm, VM: ?Sized, NN>(State<'vm, VM, NN>);

struct State<'vm, VM: ?Sized, NN> {
    stack: &'vm mut String,
    vm: &'vm VM,
    _nn: PhantomData<NN>,
}

pub trait VirtualMachineExt: VirtualMachine {
    fn init<NN: NuralNetwork<Self>>(&self, pid: pid, dev_id: dev_id, data: NN::Data) -> &Self {
        let mut stack = Domain { pid, dev_id }.to_string();

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
        dev_id: dev_id,
        nn: &NN,
        args: NN::Args<'_>,
    ) -> &Self {
        let mut stack = Domain { pid, dev_id }.to_string();

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

impl<VM: VirtualMachine> VirtualMachineExt for VM {}

impl<VM: ?Sized, NN> Context<'_, VM, NN> {
    pub fn stack(&self) -> ObjId {
        stack(self.0.stack)
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
        let len = push(self.0.stack, id);

        sub.forward(
            args,
            Context(State {
                stack: self.0.stack,
                vm: self.0.vm,
                _nn: PhantomData,
            }),
        );

        pop(self.0.stack, len);
        self
    }

    pub fn workspace(&self, dt: DigitLayout, shape: &[usize]) -> Tensor<'vm, VM> {
        let size = shape.iter().product::<usize>() * dt.nbytes() / dt.group_size();
        let blob = self.0.vm.alloc(stack(self.0.stack), size);
        Tensor::new(dt, shape, blob, self.0.vm)
    }

    pub fn get_mapped(&self, which: NN::Obj, dt: DigitLayout, shape: &[usize]) -> Tensor<'vm, VM> {
        let blob = self.0.vm.get_mapped(obj_id(self.0.stack, which));
        Tensor::new(dt, shape, blob, self.0.vm)
    }
}

impl<VM, NN> Mapping<'_, VM, NN>
where
    VM: VirtualMachine + ?Sized,
    NN: NuralNetwork<VM>,
{
    pub fn trap<Sub: NuralNetwork<VM>>(&mut self, id: NN::Sub, data: Sub::Data) -> &mut Self {
        let len = push(self.0.stack, id);

        Sub::init(
            data,
            Mapping(State {
                stack: self.0.stack,
                vm: self.0.vm,
                _nn: PhantomData,
            }),
        );

        pop(self.0.stack, len);
        self
    }

    pub fn map_host(&mut self, which: NN::Obj, mem: Box<dyn Deref<Target = [u8]>>) -> &mut Self {
        self.0
            .vm
            .free(self.0.vm.map_host(obj_id(self.0.stack, which), mem));
        self
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
        write!(f, "]Ω")
    }
}
