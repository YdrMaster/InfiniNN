use std::collections::HashMap;

#[repr(transparent)]
pub(super) struct Namespace(Vec<NameFrame>);

pub(super) struct NameFrame {
    path: String,
    pub tensor: NameDecorator,
    pub sub_nn: NameDecorator,
    pub operator: NameDecorator,
}

#[derive(Default)]
#[repr(transparent)]
pub(super) struct NameDecorator(HashMap<String, usize>);

impl Namespace {
    pub fn new(root: impl ToString) -> Self {
        Self(vec![NameFrame::new(root.to_string())])
    }

    pub fn top(&self) -> &NameFrame {
        self.0.last().unwrap()
    }

    pub fn top_mut(&mut self) -> &mut NameFrame {
        self.0.last_mut().unwrap()
    }

    pub fn push(&mut self, name: impl ToString) {
        let path = {
            let top = self.0.last_mut().unwrap();
            let name = top.sub_nn.decorate(name.to_string());
            format!("{}.{}", top.path, name)
        };
        self.0.push(NameFrame::new(path))
    }

    pub fn pop(&mut self) {
        self.0.pop();
    }
}

impl NameFrame {
    fn new(path: String) -> Self {
        Self {
            path,
            tensor: Default::default(),
            sub_nn: Default::default(),
            operator: Default::default(),
        }
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
