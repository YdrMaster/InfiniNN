use super::key_weak::KeyWeak;
use crate::{BlobLifeTime, Graph, Info};
use std::{cmp::Ordering, ops::Range};

pub struct Action<T> {
    pub i_node: usize,
    pub op: Operation,
    pub blob: KeyWeak<Info<T>>,
}

impl<T> PartialEq for Action<T> {
    fn eq(&self, other: &Self) -> bool {
        matches!(self.cmp(other), Ordering::Equal)
    }
}

impl<T> Eq for Action<T> {}

impl<T> PartialOrd for Action<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Action<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.i_node.cmp(&other.i_node) {
            Ordering::Equal => match self.op.cmp(&other.op) {
                Ordering::Equal => self.blob.cmp(&other.blob),
                ord => ord,
            },
            ord => ord,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum Operation {
    Alloc = 0,
    Free = 1,
}

impl<T> Graph<T> {
    pub fn to_actions(&self) -> Box<[Action<T>]> {
        let mut actions = self
            .blob_lifetime()
            .into_iter()
            .flat_map(|blt| {
                let BlobLifeTime { blob, life_time } = blt;
                let Range { start, end } = life_time;
                [
                    Action {
                        i_node: start,
                        op: Operation::Alloc,
                        blob: blob.clone(),
                    },
                    Action {
                        i_node: end,
                        op: Operation::Free,
                        blob,
                    },
                ]
            })
            .collect::<Box<_>>();
        actions.sort_unstable();
        actions
    }
}
