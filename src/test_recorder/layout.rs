use super::{TestTrapTracer, slice_of};
use crate::{LayoutManage, TrapTrace};
use ndarray_layout::ArrayLayout;
use patricia_tree::PatriciaMap;
use std::cell::RefCell;

#[derive(Default)]
pub(crate) struct TestLayoutManager {
    tracer: TestTrapTracer,
    layouts: RefCell<PatriciaMap<ArrayLayout<4>>>,
}

impl TrapTrace for TestLayoutManager {
    fn step_in<T: Copy>(&self, ctx: T) {
        self.tracer.step_in(ctx)
    }

    fn step_out(&self) {
        self.tracer.step_out()
    }
}

impl LayoutManage for TestLayoutManager {
    fn get<T: Copy>(&self, which: T) -> ArrayLayout<4> {
        let mut key = self.tracer.current();
        key.extend_from_slice(slice_of(&which));

        self.layouts.borrow().get(&key).unwrap().clone()
    }

    fn set<T: Copy>(&self, which: T, layout: ArrayLayout<4>) {
        let mut key = self.tracer.current();
        key.extend_from_slice(slice_of(&which));

        let mut map = self.layouts.borrow_mut();
        map.insert(key, layout);
    }
}
