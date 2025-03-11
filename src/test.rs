#![cfg(test)]

use crate::{ObjId, VirtualMachine, pid};
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

impl crate::CommGroup for CommGroup {
    fn n_members(&self) -> usize {
        self.0
    }
}

#[derive(Default)]
struct Internal {
    next_pid: pid,
    next_blob_id: usize,
    blobs: PatriciaMap<usize>,
    bcbs: HashMap<usize, Bcb>,
}

/// Bcb for Blob Control Block
struct Bcb {
    id: usize,
    n_bytes: usize,
    ref_count: usize,
    obj_id: ObjId,
    _ty: BlobType,
}

impl Bcb {
    fn retain_blob(&mut self) -> Blob {
        self.ref_count += 1;
        Blob {
            id: self.id,
            n_bytes: self.n_bytes,
        }
    }
}

enum BlobType {
    Host,
    Device,
    #[allow(unused)]
    Map(Box<dyn Deref<Target = [u8]>>),
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
        self.alloc_(obj, mem.len(), BlobType::Map(mem))
    }

    fn get_mapped(&self, obj: ObjId) -> Self::Blob {
        let mut internal = self.0.borrow_mut();

        let id = *internal.blobs.get(obj.as_slice()).unwrap();
        println!("[{}] load %{id}", obj.domain());
        internal.bcbs.get_mut(&id).unwrap().retain_blob()
    }

    fn alloc_host(&self, obj: ObjId, size: usize) -> Self::Blob {
        self.alloc_(obj, size, BlobType::Host)
    }

    fn alloc(&self, obj: ObjId, size: usize) -> Self::Blob {
        self.alloc_(obj, size, BlobType::Device)
    }

    fn retain(&self, obj: &Self::Blob) -> Self::Blob {
        let mut internal = self.0.borrow_mut();

        let bcb = internal.bcbs.get_mut(&obj.id).unwrap();

        println!("[{}] retain %{}", bcb.obj_id.domain(), bcb.id);

        bcb.retain_blob()
    }

    fn release(&self, blob: Self::Blob) {
        let mut internal = self.0.borrow_mut();

        let bcb = internal.bcbs.get_mut(&blob.id).unwrap();
        bcb.ref_count -= 1;

        let id = bcb.id;

        let domain = bcb.obj_id.domain();

        if bcb.ref_count == 0 {
            let bcb = internal.bcbs.remove(&id).unwrap();
            internal.blobs.remove(bcb.obj_id.as_slice());
            println!("[{domain}] drop %{}", bcb.id)
        } else {
            println!("[{domain}] release %{}", bcb.id)
        }
    }

    fn comm(&self, devices: &[usize]) -> Self::CommGroup {
        CommGroup(devices.len())
    }
}

impl TestVM {
    pub(crate) fn launch(&self, obj: ObjId, info: String) {
        println!("[{}] {info} @ {}", obj.domain(), obj.body())
    }

    fn alloc_(&self, obj: ObjId, size: usize, ty: BlobType) -> Blob {
        let mut internal = self.0.borrow_mut();

        let id = internal.next_blob_id;
        internal.next_blob_id += 1;

        if !obj.is_free() {
            assert!(internal.blobs.insert(obj.as_slice(), id).is_none())
        } else {
            assert!(matches!(ty, BlobType::Host | BlobType::Device))
        }

        let domain = obj.domain();
        let body = obj.body();
        match &ty {
            BlobType::Host => println!("[{domain}] alloc %{id} {size} bytes @ {body}"),
            BlobType::Device => println!("[{domain}] alloc %{id} {size} bytes @ {body}"),
            BlobType::Map(_) => println!("[{domain}] map %{id} {size} bytes @ {body}"),
        }

        let mut bcb = Bcb {
            id,
            n_bytes: size,
            ref_count: 0,
            obj_id: obj,
            _ty: ty,
        };
        let ans = bcb.retain_blob();
        internal.bcbs.insert(id, bcb);
        ans
    }
}
