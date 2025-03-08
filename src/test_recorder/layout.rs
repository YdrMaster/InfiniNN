use super::TestTrapTracer;
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
        let key = self.tracer.leaf(which);
        self.layouts.borrow().get(&key).unwrap().clone()
    }

    fn set<T: Copy>(&self, which: T, layout: ArrayLayout<4>) {
        let key = self.tracer.leaf(which);
        let mut map = self.layouts.borrow_mut();
        map.insert(key, layout);
    }
}

impl<T, U> From<T> for TestLayoutManager
where
    U: Copy,
    T: IntoIterator<Item = (U, ArrayLayout<4>)>,
{
    fn from(value: T) -> Self {
        let ans = Self::default();
        for (which, layout) in value {
            ans.set(which, layout)
        }
        ans
    }
}

impl<U: Copy> Extend<(U, ArrayLayout<4>)> for TestLayoutManager {
    fn extend<T: IntoIterator<Item = (U, ArrayLayout<4>)>>(&mut self, iter: T) {
        for (which, layout) in iter {
            self.set(which, layout)
        }
    }
}
