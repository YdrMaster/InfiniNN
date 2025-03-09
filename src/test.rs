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

impl crate::Blob for Blob {
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
    blob_map: HashMap<usize, BCB>,
}

struct BCB {
    id: usize,
    n_bytes: usize,
    ref_count: usize,
    obj_id: ObjId,
    _placement: Placement,
    _mem: Option<Box<dyn Deref<Target = [u8]>>>,
}

impl BCB {
    fn retain_blob(&mut self) -> Blob {
        self.ref_count += 1;
        Blob {
            id: self.id,
            n_bytes: self.n_bytes,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Placement {
    Host,
    Device,
}

impl VirtualMachine for TestVM {
    type Blob = Blob;
    type CommGroup = CommGroup;

    fn num_devices(&self) -> usize {
        1
    }

    fn register(&self, _arch: &str) -> pid {
        let mut internal = self.0.borrow_mut();
        let pid = internal.next_pid;
        internal.next_pid += 1;
        pid
    }

    fn unregister(&self, _pid: pid) {}

    fn map_host(&self, obj: ObjId, mem: Box<dyn Deref<Target = [u8]>>) -> Self::Blob {
        let mut internal = self.0.borrow_mut();

        let id = internal.next_blob_id;
        internal.next_blob_id += 1;

        internal.blobs.insert(obj.as_slice(), id);

        let mut bcb = BCB {
            id,
            n_bytes: mem.len(),
            ref_count: 0,
            obj_id: obj,
            _placement: Placement::Device,
            _mem: Some(mem),
        };
        let ans = bcb.retain_blob();
        internal.blob_map.insert(id, bcb);
        ans
    }

    fn get_mapped(&self, obj: ObjId) -> Self::Blob {
        let mut internal = self.0.borrow_mut();

        let id = *internal.blobs.get(obj.as_slice()).unwrap();
        internal.blob_map.get_mut(&id).unwrap().retain_blob()
    }

    fn alloc_host(&self, obj: ObjId, size: usize) -> Self::Blob {
        let mut internal = self.0.borrow_mut();

        let id = internal.next_blob_id;
        internal.next_blob_id += 1;

        internal.blobs.insert(obj.as_slice(), id);

        let mut bcb = BCB {
            id,
            n_bytes: size,
            ref_count: 1,
            obj_id: obj,
            _placement: Placement::Host,
            _mem: None,
        };
        let ans = bcb.retain_blob();
        internal.blob_map.insert(id, bcb);
        ans
    }

    fn alloc(&self, obj: ObjId, size: usize) -> Self::Blob {
        let mut internal = self.0.borrow_mut();

        let id = internal.next_blob_id;
        internal.next_blob_id += 1;

        internal.blobs.insert(obj.as_slice(), id);

        let mut bcb = BCB {
            id,
            n_bytes: size,
            ref_count: 0,
            obj_id: obj,
            _placement: Placement::Device,
            _mem: None,
        };
        let ans = bcb.retain_blob();
        internal.blob_map.insert(id, bcb);
        ans
    }

    fn retain(&self, obj: &Self::Blob) -> Self::Blob {
        let mut internal = self.0.borrow_mut();

        internal.blob_map.get_mut(&obj.id).unwrap().retain_blob()
    }

    fn release(&self, blob: Self::Blob) {
        let mut internal = self.0.borrow_mut();

        let bcb = internal.blob_map.get_mut(&blob.id).unwrap();
        bcb.ref_count -= 1;

        let id = bcb.id;
        if bcb.ref_count == 0 {
            let bcb = internal.blob_map.remove(&id).unwrap();
            internal.blobs.remove(bcb.obj_id.as_slice());
        }
    }

    fn comm(&self, devices: &[usize]) -> Self::CommGroup {
        CommGroup(devices.len())
    }
}
