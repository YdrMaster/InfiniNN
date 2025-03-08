#![cfg(test)]

mod layout;
mod memory;

pub(crate) use layout::TestLayoutManager;
pub(crate) use memory::{TestMemManager, TestMemManagerLoader};

use crate::TrapTrace;
use std::cell::RefCell;

#[derive(Default)]
#[repr(transparent)]
struct TestTrapTracer(RefCell<Vec<u8>>);

impl TestTrapTracer {
    pub fn leaf(&self, key: impl Copy) -> Vec<u8> {
        let slice = slice_of(&key);
        assert!(slice.len() < 256);

        let mut ans = self.0.borrow().clone();
        ans.push(slice.len() as _);
        ans.extend_from_slice(slice);
        ans
    }
}

impl TrapTrace for TestTrapTracer {
    fn step_in<T: Copy>(&self, ctx: T) {
        let slice = slice_of(&ctx);
        assert!(slice.len() < 256);

        let mut vec = self.0.borrow_mut();
        vec.push(slice.len() as _);
        vec.extend_from_slice(slice);
        vec.push(slice.len() as _);
    }

    fn step_out(&self) {
        let mut vec = self.0.borrow_mut();
        let Some(len) = vec.pop() else {
            return;
        };
        let len = vec.len() - (len as usize + 1);
        vec.truncate(len)
    }
}

#[inline(always)]
fn slice_of(val: &impl Copy) -> &[u8] {
    unsafe { std::slice::from_raw_parts((&raw const *val).cast(), size_of_val(val)) }
}
