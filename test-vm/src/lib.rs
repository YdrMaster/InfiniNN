mod op;

use digit_layout::DigitLayout;
use patricia_tree::PatriciaMap;
use std::{cell::RefCell, collections::HashMap, ops::Deref};
use vm::{Id, ObjId, VirtualMachine, pid};

#[derive(Default)]
#[repr(transparent)]
pub struct TestVM(RefCell<Internal>);

pub struct Blob {
    id: usize,
    n_bytes: usize,
}

impl Blob {
    pub fn id(&self) -> usize {
        self.id
    }
}

impl vm::Blob for Blob {
    fn eq(l: &Self, r: &Self) -> bool {
        l.id == r.id
    }

    fn n_bytes(&self) -> usize {
        self.n_bytes
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CommGroup(usize);

impl Id for CommGroup {
    fn name(&self) -> &str {
        "test-comm"
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

#[derive(Default)]
struct Internal {
    next_pid: pid,
    next_blob_id: usize,
    maps: PatriciaMap<usize>,
    bcbs: HashMap<usize, Bcb>,
}

/// Bcb for Blob Control Block
struct Bcb {
    id: usize,
    n_bytes: usize,
    obj: ObjId,
    rc: usize,
}

impl VirtualMachine for TestVM {
    type Blob = Blob;
    type CommGroup = CommGroup;

    fn register(&self, arch: &str) -> pid {
        let mut internal = self.0.borrow_mut();
        let pid = internal.next_pid;
        internal.next_pid += 1;
        println!("[vm:_] register {arch} -> #{pid:x}");
        pid
    }

    fn unregister(&self, pid: pid) {
        println!("[vm:_] unregister #{pid:x}")
    }

    fn map_host(&self, obj: ObjId, mem: Box<dyn Deref<Target = [u8]>>) -> Self::Blob {
        self.alloc_(obj.clone(), mem.len())
    }

    fn get_mapped(&self, obj: ObjId) -> Self::Blob {
        let mut internal = self.0.borrow_mut();

        let id = *internal.maps.get_mut(obj.as_str()).unwrap();

        println!("{} load %{id} @ {}", obj.domain(), obj.body());

        let bcb = internal.bcbs.get_mut(&id).unwrap();
        bcb.rc += 1;
        Blob {
            id: bcb.id,
            n_bytes: bcb.n_bytes,
        }
    }

    fn alloc(&self, obj: ObjId, size: usize) -> Self::Blob {
        self.alloc_(obj, size)
    }

    fn free(&self, blob: Self::Blob) {
        let mut internal = self.0.borrow_mut();
        let Internal { maps, bcbs, .. } = &mut *internal;

        let bcb = bcbs.get_mut(&blob.id).unwrap();
        bcb.rc -= 1;
        if bcb.rc == 0 {
            println!("{} free %{}", bcb.obj.domain(), bcb.id);
            if !maps.contains_key(bcb.obj.as_str()) {
                bcbs.remove(&blob.id).unwrap();
            }
        }
    }

    fn comm(&self, devices: &[usize]) -> Self::CommGroup {
        CommGroup(devices.len())
    }
}

impl TestVM {
    pub fn launch(&self, obj: ObjId, info: String) {
        println!("{} {info} @ {}", obj.domain(), obj.body())
    }

    fn alloc_(&self, obj: ObjId, size: usize) -> Blob {
        let mut internal = self.0.borrow_mut();

        let id = internal.next_blob_id;
        internal.next_blob_id += 1;

        let domain = obj.domain();
        let body = obj.body();
        println!("{domain} alloc %{id} {size} bytes @ {body}");

        if obj.is_obj() {
            assert!(internal.maps.insert(obj.as_str(), id).is_none())
        }

        let bcb = Bcb {
            id,
            obj,
            n_bytes: size,
            rc: 1,
        };
        internal.bcbs.insert(id, bcb);
        Blob { id, n_bytes: size }
    }
}

pub fn test_data(dt: DigitLayout, shape: &[usize]) -> Box<dyn Deref<Target = [u8]>> {
    let size = shape.iter().product::<usize>() * dt.nbytes() / dt.group_size();
    Box::new(vec![0u8; size])
}
