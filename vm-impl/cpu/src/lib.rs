mod fmt;
mod op;

use patricia_tree::PatriciaMap;
use std::{
    borrow::Cow,
    ops::Deref,
    sync::{
        Arc, RwLock,
        atomic::{AtomicU64, Ordering::SeqCst},
    },
};
use vm::{Id, ObjId, VirtualMachine, pid};

#[derive(Default)]
pub struct CpuVM {
    next_pid: AtomicU64,
    maps: RwLock<PatriciaMap<SharedSlice>>,
}

impl VirtualMachine for CpuVM {
    type Blob = Blob;
    type CommGroup = CommGroup;

    fn register(&self, _arch: &str) -> pid {
        self.next_pid.fetch_add(1, SeqCst)
    }

    fn unregister(&self, _pid: pid) {}

    fn map_host(&self, obj: ObjId, mem: Box<dyn Deref<Target = [u8]>>) -> Self::Blob {
        if obj.is_obj() {
            let mem: SharedSlice = mem.into();
            assert!(
                self.maps
                    .write()
                    .unwrap()
                    .insert(&obj, mem.clone())
                    .is_none()
            );
            Blob::Mapped(mem)
        } else {
            Blob::Allocated(mem.to_vec().into())
        }
    }

    fn get_mapped(&self, obj: ObjId) -> Self::Blob {
        assert!(obj.is_obj());
        Blob::Mapped(self.maps.read().unwrap().get(&obj).unwrap().clone())
    }

    fn alloc(&self, _obj: ObjId, size: usize) -> Self::Blob {
        Blob::Allocated(vec![0u8; size].into())
    }

    fn free(&self, _blob: Self::Blob) {}

    fn comm(&self, _devices: &[usize]) -> Self::CommGroup {
        todo!()
    }
}

type SharedSlice = Arc<dyn Deref<Target = [u8]>>;

pub enum Blob {
    Mapped(SharedSlice),
    Allocated(Box<[u8]>),
}

impl Deref for Blob {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        match self {
            Blob::Mapped(m) => m,
            Blob::Allocated(m) => m,
        }
    }
}

impl vm::Blob for Blob {
    fn eq(l: &Self, r: &Self) -> bool {
        l.as_ptr() == r.as_ptr()
    }

    fn n_bytes(&self) -> usize {
        match self {
            Blob::Mapped(m) => m.len(),
            Blob::Allocated(m) => m.len(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CommGroup(usize);

impl Id for CommGroup {
    fn name(&self) -> Cow<str> {
        "test-comm".into()
    }

    fn idx(&self) -> Option<usize> {
        Some(self.0)
    }
}

impl vm::CommGroup for CommGroup {
    fn n_members(&self) -> usize {
        self.0
    }
}

#[cfg(test)]
mod test_llama;
