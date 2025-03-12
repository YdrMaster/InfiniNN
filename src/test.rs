#![cfg(test)]

use crate::{Id, ObjId, VirtualMachine, pid};
use patricia_tree::PatriciaMap;
use std::{cell::RefCell, collections::HashMap, ops::Deref};

#[derive(Default)]
#[repr(transparent)]
pub(crate) struct TestVM(RefCell<Internal>);

pub(crate) struct Blob {
    id: usize,
    n_bytes: usize,
}

impl Blob {
    pub fn id(&self) -> usize {
        self.id
    }
}

impl crate::Blob for Blob {
    fn eq(l: &Self, r: &Self) -> bool {
        l.id == r.id
    }

    fn n_bytes(&self) -> usize {
        self.n_bytes
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct CommGroup(usize);

impl Id for CommGroup {
    fn name(&self) -> &str {
        "test-comm"
    }

    fn idx(&self) -> Option<usize> {
        Some(self.0)
    }
}

impl crate::CommGroup for CommGroup {
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
    _ty: Placement,
}

enum Placement {
    Host,
    Device,
    Mapped {
        rc: usize,
        _map: Box<dyn Deref<Target = [u8]>>,
    },
}

impl VirtualMachine for TestVM {
    type Blob = Blob;
    type CommGroup = CommGroup;

    fn num_devices(&self) -> usize {
        1
    }

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
        self.alloc_(
            obj.clone(),
            mem.len(),
            Placement::Mapped { rc: 1, _map: mem },
        )
    }

    fn get_mapped(&self, obj: ObjId) -> Self::Blob {
        let mut internal = self.0.borrow_mut();

        let id = *internal.maps.get_mut(obj.as_str()).unwrap();

        println!("{} load %{id} @ {}", obj.domain(), obj.body());

        let bcb = internal.bcbs.get_mut(&id).unwrap();
        match &mut bcb._ty {
            Placement::Mapped { rc, .. } => *rc += 1,
            Placement::Host | Placement::Device => unreachable!(),
        }
        Blob {
            id: bcb.id,
            n_bytes: bcb.n_bytes,
        }
    }

    fn alloc_host(&self, obj: ObjId, size: usize) -> Self::Blob {
        self.alloc_(obj, size, Placement::Host)
    }

    fn alloc(&self, obj: ObjId, size: usize) -> Self::Blob {
        self.alloc_(obj, size, Placement::Device)
    }

    fn free(&self, blob: Self::Blob) {
        let mut internal = self.0.borrow_mut();

        let bcb = internal.bcbs.get_mut(&blob.id).unwrap();
        match &mut bcb._ty {
            Placement::Host | Placement::Device => {
                println!("{} free %{}", bcb.obj.domain(), bcb.id);
                internal.bcbs.remove(&blob.id);
            }
            Placement::Mapped { rc, .. } => {
                *rc -= 1;
                if *rc == 0 {
                    println!("{} free %{}", bcb.obj.domain(), bcb.id)
                }
            }
        }
    }

    fn comm(&self, devices: &[usize]) -> Self::CommGroup {
        CommGroup(devices.len())
    }
}

impl TestVM {
    pub(crate) fn launch(&self, obj: ObjId, info: String) {
        println!("{} {info} @ {}", obj.domain(), obj.body())
    }

    fn alloc_(&self, obj: ObjId, size: usize, placement: Placement) -> Blob {
        let mut internal = self.0.borrow_mut();

        let id = internal.next_blob_id;
        internal.next_blob_id += 1;

        let domain = obj.domain();
        let body = obj.body();
        match &placement {
            Placement::Host => println!("{domain} alloc %{id} {size} bytes @ {body}"),
            Placement::Device => println!("{domain} alloc %{id} {size} bytes @ {body}"),
            Placement::Mapped { .. } => {
                println!("{domain} map %{id} {size} bytes @ {body}");
                assert!(internal.maps.insert(obj.as_str(), id).is_none());
            }
        }

        let bcb = Bcb {
            id,
            obj,
            n_bytes: size,
            _ty: placement,
        };
        internal.bcbs.insert(id, bcb);
        Blob { id, n_bytes: size }
    }
}
