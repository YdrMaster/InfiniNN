use std::{cell::RefCell, collections::HashMap, rc::Rc};

#[derive(Clone)]
#[repr(transparent)]
pub(super) struct Stack(Rc<RefCell<Vec<Rc<RefCell<Frame>>>>>);

pub(super) struct Frame {
    path: String,
    pub tensor: NameDecorator,
    pub sub_nn: NameDecorator,
    pub operator: NameDecorator,
}

#[derive(Default)]
#[repr(transparent)]
pub(super) struct NameDecorator(HashMap<String, usize>);

impl Stack {
    pub fn new(root: impl ToString) -> Self {
        Self(Rc::new(RefCell::new(vec![Frame::new(root.to_string())])))
    }

    pub fn top(&self) -> Rc<RefCell<Frame>> {
        self.0.borrow().last().unwrap().clone()
    }

    pub fn push(&self, name: impl ToString) {
        let mut stack = self.0.borrow_mut();
        let path = {
            let mut top = stack.last_mut().unwrap().borrow_mut();
            let name = top.sub_nn.decorate(name.to_string());
            format!("{}.{}", top.path, name)
        };
        stack.push(Frame::new(path))
    }

    pub fn pop(&self) {
        self.0.borrow_mut().pop();
    }
}

impl Frame {
    fn new(path: String) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            path,
            tensor: Default::default(),
            sub_nn: Default::default(),
            operator: Default::default(),
        }))
    }

    pub fn path(&self) -> &str {
        &self.path
    }
}

impl NameDecorator {
    pub fn check(&mut self, name: impl ToString) -> bool {
        use std::collections::hash_map::Entry::*;
        match self.0.entry(name.to_string()) {
            Vacant(entry) => {
                entry.insert(1);
                true
            }
            Occupied(_) => false,
        }
    }

    pub fn decorate(&mut self, name: String) -> String {
        use std::collections::hash_map::Entry::*;
        match self.0.entry(name) {
            Occupied(mut entry) => {
                *entry.get_mut() += 1;
                format!("{}-{}", entry.key(), entry.get())
            }
            Vacant(entry) => {
                let ans = entry.key().clone();
                entry.insert(1);
                ans
            }
        }
    }
}
