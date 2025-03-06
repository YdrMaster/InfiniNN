use super::{TestTrapTracer, slice_of};
use crate::{Backend, MemManage, Ptr, TrapTrace};
use patricia_tree::PatriciaMap;
use std::{cell::RefCell, ffi::c_void, fmt};

pub(crate) struct TestRecorder;

impl Backend for TestRecorder {
    type Byte = c_void;
}

#[derive(Default)]
pub(crate) struct TestMemManager {
    tracer: TestTrapTracer,
    internal: RefCell<Internal>,
}

#[derive(Default)]
struct Internal {
    args: PatriciaMap<Ptr<TestRecorder>>,
    regs: Vec<Reg>,
    records: Vec<Record>,
}

enum Reg {
    Arg(Box<[u8]>),
    Workspace(usize),
}

#[derive(Clone)]
pub(crate) enum Record {
    Load(Ptr<TestRecorder>, bool),
    Drop(Ptr<TestRecorder>),
    Launch(String),
}

impl TrapTrace for TestMemManager {
    fn step_in<T: Copy>(&self, ctx: T) {
        self.tracer.step_in(ctx)
    }

    fn step_out(&self) {
        self.tracer.step_out()
    }
}

impl MemManage for TestMemManager {
    type B = TestRecorder;

    fn push_arg<T: Copy>(&self, which: T, ptr: Ptr<Self::B>) {
        let mut key = self.tracer.current();
        key.extend_from_slice(slice_of(&which));

        let mut internal = self.internal.borrow_mut();
        assert!(internal.args.insert(key, ptr).is_none())
    }

    fn pop_arg<T: Copy>(&self, which: T) {
        let mut key = self.tracer.current();
        key.extend_from_slice(slice_of(&which));

        let mut internal = self.internal.borrow_mut();
        assert!(internal.args.remove(key).is_some())
    }

    fn malloc(&self, size: usize) -> Ptr<Self::B> {
        let mut internal = self.internal.borrow_mut();
        let reg = Ptr::Mut(internal.regs.len() as _);
        internal.regs.push(Reg::Workspace(size));
        internal.records.push(Record::Load(reg, true));
        reg
    }

    fn load<T: Copy>(&self, which: T, mutable: bool) -> Ptr<Self::B> {
        let mut key = self.tracer.current();
        key.extend_from_slice(slice_of(&which));

        let mut internal = self.internal.borrow_mut();
        let &reg = internal.args.get(&key).expect("arg not exist");
        internal.records.push(Record::Load(reg, mutable));
        reg
    }

    fn drop(&self, ptr: Ptr<Self::B>) {
        self.internal.borrow_mut().records.push(Record::Drop(ptr))
    }
}

impl TestMemManager {
    pub fn launch(&self, info: String) {
        self.internal
            .borrow_mut()
            .records
            .push(Record::Launch(info))
    }

    pub fn put_arg(&self, which: impl Copy) {
        let mut key = self.tracer.current();
        key.extend_from_slice(slice_of(&which));

        let mut internal = self.internal.borrow_mut();
        let reg = Ptr::Mut(internal.regs.len() as _);
        internal.regs.push(Reg::Arg(key.clone().into_boxed_slice()));
        assert!(internal.args.insert(key, reg).is_none())
    }
}

impl fmt::Display for TestMemManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let internal = self.internal.borrow();
        let wreg = internal.regs.len() / 10 + 1;
        for (i, reg) in internal.regs.iter().enumerate() {
            write!(f, "%{i:0wreg$} ")?;
            match reg {
                Reg::Arg(path) => {
                    write!(f, "\"")?;
                    for byte in path {
                        write!(f, "{:02x}", byte)?;
                    }
                    writeln!(f, "\"")?
                }
                Reg::Workspace(size) => writeln!(f, "[{size}]")?,
            }
        }
        writeln!(f)?;
        writeln!(f, "---")?;
        writeln!(f)?;
        let wrec = internal.records.len() / 10 + 1;
        for (i, record) in internal.records.iter().enumerate() {
            write!(f, "{i:0wrec$} ")?;
            match record {
                Record::Load(reg, true) => writeln!(f, "ld mut %{:0wreg$}", reg.address()),
                Record::Load(reg, false) => writeln!(f, "ld ref %{:0wreg$}", reg.address()),
                Record::Drop(reg) => writeln!(f, "drop   %{:0wreg$}", reg.address()),
                Record::Launch(info) => writeln!(f, "call   {info}"),
            }?
        }
        Ok(())
    }
}
