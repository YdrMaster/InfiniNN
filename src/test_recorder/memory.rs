use super::TestTrapTracer;
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

#[derive(Clone)]
enum Reg {
    Arg(Vec<u8>),
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
        let key = self.tracer.leaf(which);
        let mut internal = self.internal.borrow_mut();
        assert!(internal.args.insert(key, ptr).is_none())
    }

    fn pop_arg<T: Copy>(&self, which: T) {
        let key = self.tracer.leaf(which);
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
        let key = self.tracer.leaf(which);
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
}

impl fmt::Display for TestMemManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let internal = self.internal.borrow();
        let wreg = internal.regs.len() / 10 + 1;
        for (i, reg) in internal.regs.iter().enumerate() {
            write!(f, "%{i:0wreg$} ")?;
            match reg {
                Reg::Arg(key) => {
                    let mut key = &**key;
                    let mut vec = Vec::new();
                    while let [len, tail @ ..] = key {
                        let len = *len as usize;
                        vec.push(&tail[..len]);
                        if let [len_, tail @ ..] = &tail[len..] {
                            assert_eq!(len, *len_ as _);
                            key = tail
                        } else {
                            break;
                        }
                    }
                    writeln!(f, "{vec:x?}")?
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

pub(crate) struct TestMemManagerLoader {
    tracer: TestTrapTracer,
    args: PatriciaMap<Ptr<TestRecorder>>,
}

impl TestMemManagerLoader {
    pub fn new<T: Copy>(
        r#mut: impl IntoIterator<Item = T>,
        r#const: impl IntoIterator<Item = T>,
    ) -> Self {
        let mut ans = Self {
            tracer: Default::default(),
            args: Default::default(),
        };
        for which in r#mut {
            let key = ans.tracer.leaf(which);
            let reg = ans.args.len();
            ans.args.insert(key, Ptr::Mut(reg as _));
        }
        for which in r#const {
            let key = ans.tracer.leaf(which);
            let reg = ans.args.len();
            ans.args.insert(key, Ptr::Const(reg as _));
        }
        ans
    }

    #[allow(unused)]
    pub fn load_weights(&mut self) -> LoaderState {
        LoaderState(self)
    }

    pub fn build(self) -> TestMemManager {
        let Self { args, .. } = self;
        let mut regs = vec![Reg::Arg(vec![]); args.len()];
        for (key, ptr) in args.iter() {
            regs[ptr.address() as usize] = Reg::Arg(key)
        }
        TestMemManager {
            internal: RefCell::new(Internal {
                args,
                regs,
                ..Default::default()
            }),
            ..Default::default()
        }
    }
}

pub(crate) struct LoaderState<'a>(&'a mut TestMemManagerLoader);

impl LoaderState<'_> {
    #[allow(unused)]
    pub fn trap_with<T, U, I>(&mut self, trap: T, ptrs: I) -> LoaderState
    where
        T: Copy,
        U: Copy,
        I: IntoIterator<Item = U>,
    {
        self.0.tracer.step_in(trap);
        for which in ptrs {
            let key = self.0.tracer.leaf(which);
            let reg = self.0.args.len();
            self.0.args.insert(key, Ptr::Const(reg as _));
        }
        LoaderState(self.0)
    }
}

impl Drop for LoaderState<'_> {
    fn drop(&mut self) {
        self.0.tracer.step_out()
    }
}
