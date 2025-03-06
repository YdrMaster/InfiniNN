use crate::{Backend, LayoutManager, MemManager, MemManagerExt, StorageTensor, Tensor};
use digit_layout::DigitLayout;

pub enum Activation {
    SwiGLU { gate: Tensor, up: Tensor },
    GeLU { up: Tensor },
}

#[derive(Clone, Copy, Debug)]
pub struct Meta {
    pub ty: Type,
    pub dt: DigitLayout,
    pub di: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum Type {
    SwiGLU,
    GeLU,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Arg {
    Gate,
    Up,
}

impl Meta {
    pub fn build(&self, env: &impl LayoutManager<Arg>, batch_size: usize) -> Activation {
        let &Self { ty, dt, di } = self;
        let shape = [batch_size, di];
        match ty {
            Type::SwiGLU => Activation::SwiGLU {
                gate: env.tensor(Arg::Gate, dt, &shape),
                up: env.tensor(Arg::Up, dt, &shape),
            },
            Type::GeLU => Activation::GeLU {
                up: env.tensor(Arg::Up, dt, &shape),
            },
        }
    }
}

pub trait Env<B: Backend>: MemManager<Arg, B> {
    fn swiglu(&self, gate: &mut StorageTensor, up: &StorageTensor);
    fn gelu(&self, up: &mut StorageTensor);
}

impl Activation {
    pub fn launch<B: Backend>(&self, env: &impl Env<B>) {
        match self {
            Self::SwiGLU { gate, up } => {
                let mut gate = env.load_tensor_mut(Arg::Gate, gate);
                let up = env.load_tensor_mut(Arg::Up, up);
                env.swiglu(&mut gate, &up)
            }
            Self::GeLU { up } => {
                let mut up = env.load_tensor_mut(Arg::Up, up);
                env.gelu(&mut up)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Arg, Env, Meta, Type};
    use crate::{Backend, LayoutManager, MemManager, StorageTensor};
    use digit_layout::types as ty;
    use std::{cell::RefCell, collections::HashMap, os::raw::c_void, rc::Rc};

    struct LM(HashMap<Arg, (Vec<isize>, isize)>);

    impl LayoutManager<Arg> for LM {
        fn get(&self, which: Arg) -> (&[isize], isize) {
            let (ref layout, offset) = self.0[&which];
            (layout, offset)
        }

        fn set(&mut self, which: Arg, layout: (&[isize], isize)) {
            self.0.insert(which, (layout.0.to_vec(), layout.1));
        }
    }

    struct MM(Rc<RefCell<Internal>>);

    struct Internal {
        idx: usize,
        mem: HashMap<usize, Mem>,
        args: HashMap<Arg, usize>,
        records: Vec<Record>,
    }

    #[derive(Debug)]
    enum Mem {
        Arg(Arg),
        Workspace(usize),
    }

    #[derive(Debug)]
    enum Record {
        Load(Mem),
        Drop(Mem),
        Launch(String),
    }

    struct Test;

    impl Backend for Test {
        type Byte = c_void;
    }

    impl MemManager<Arg, Test> for MM {
        fn malloc(&self, size: usize) -> *mut <Test as Backend>::Byte {
            let mut mm = self.0.borrow_mut();
            let idx = mm.idx;
            mm.idx += 1;
            mm.mem.insert(idx, Mem::Workspace(size));
            mm.records.push(Record::Load(Mem::Workspace(size)));
            idx as _
        }

        fn load_mut(&self, which: Arg) -> *mut <Test as Backend>::Byte {
            let mut mm = self.0.borrow_mut();
            let mem = *mm.args.get(&which).expect("arg not found");
            assert!(
                mm.mem.insert(mem, Mem::Arg(which)).is_none(),
                "{which:?} already loaded"
            );
            mm.records.push(Record::Load(Mem::Arg(which)));
            mem as _
        }

        fn load(&self, which: Arg) -> *const <Test as Backend>::Byte {
            let mut mm = self.0.borrow_mut();
            let mem = *mm.args.get(&which).expect("arg not found");
            assert!(
                mm.mem.insert(mem, Mem::Arg(which)).is_none(),
                "{which:?} already loaded"
            );
            mm.records.push(Record::Load(Mem::Arg(which)));
            mem as _
        }

        fn drop(&self, ptr: *const <Test as Backend>::Byte) {
            let mut mm = self.0.borrow_mut();
            match mm.mem.remove(&(ptr as _)) {
                Some(Mem::Workspace(size)) => mm.records.push(Record::Drop(Mem::Workspace(size))),
                Some(Mem::Arg(which)) => mm.records.push(Record::Drop(Mem::Arg(which))),
                None => panic!("invalid ptr"),
            };
        }
    }

    impl Env<Test> for MM {
        fn swiglu(&self, gate: &mut StorageTensor, up: &StorageTensor) {
            let mut mm = self.0.borrow_mut();
            mm.records.push(Record::Launch("SwiGLU".to_string()));
        }

        fn gelu(&self, up: &mut StorageTensor) {
            let mut mm = self.0.borrow_mut();
            mm.records.push(Record::Launch("GeLU".to_string()));
        }
    }

    #[test]
    fn test() {
        let meta = Meta {
            ty: Type::SwiGLU,
            dt: ty::F16,
            di: 2048,
        };

        let mut lm = LM(HashMap::new());
        lm.set(Arg::Gate, (&[8192, 2], 0));
        lm.set(Arg::Up, (&[8192, 2], 8192 * 7));
        let act = meta.build(&mut lm, 7);

        let mm = MM(Rc::new(RefCell::new(Internal {
            idx: 2,
            mem: HashMap::new(),
            args: HashMap::from([(Arg::Gate, 0), (Arg::Up, 1)]),
            records: Vec::new(),
        })));

        act.launch(&mm);

        println!("{:x?}", mm.0.borrow().records);
    }
}
