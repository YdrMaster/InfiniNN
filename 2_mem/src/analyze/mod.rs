mod calculator;
mod key_weak;

use crate::{Graph, Info};
use calculator::OffsetCalculator;
use graph::NodeRef;
use key_weak::KeyWeak;
use std::{collections::HashMap, iter::zip, ops::Range, rc::Rc, usize};

impl<T> Graph<T> {
    pub fn analyze(&self, _alignment: usize) -> Box<[(KeyWeak<Info<T>>, Range<usize>)]> {
        let Self(graph::Graph { topo, nodes, edges }) = self;
        // 在分析器中标记全图输入输出
        let mut analyzer = BlobAnalyzer::new(
            topo.global_inputs().map(|i| edges[i].0.get()),
            topo.global_outputs().iter().map(|&i| edges[i].0.get()),
        );
        // 根据算子生成每个 Blob 的生命周期
        for (i, (topo, node)) in zip(topo.iter(), nodes).enumerate() {
            if node.op == "empty" {
                continue;
            }
            let NodeRef { inputs, outputs } = topo;
            for &input in inputs {
                analyzer.push(i, edges[input].0.get())
            }
            for output in outputs {
                analyzer.push(i, edges[output].0.get())
            }
        }
        analyzer.take_sorted()
    }
}

#[derive(Default)]
#[repr(transparent)]
struct BlobAnalyzer<T>(HashMap<KeyWeak<Info<T>>, Range<usize>>);

impl<T> BlobAnalyzer<T> {
    pub fn new<'a>(
        inputs: impl IntoIterator<Item = &'a Rc<Info<T>>>,
        outputs: impl IntoIterator<Item = &'a Rc<Info<T>>>,
    ) -> Self
    where
        T: 'a,
    {
        let mut ans = Self(HashMap::new());
        for blob in inputs {
            if let Some(record) = ans.get_or_insert(blob) {
                record.start = 0
            }
        }
        for blob in outputs {
            if let Some(record) = ans.get_or_insert(blob) {
                record.end = usize::MAX
            }
        }
        ans
    }

    pub fn push(&mut self, i_node: usize, blob: &Rc<Info<T>>) {
        if let Some(record) = self.get_or_insert(blob) {
            record.start = record.start.min(i_node);
            record.end = record.end.max(i_node);
        }
    }

    #[allow(unused)]
    pub fn take_sorted(self) -> Box<[(KeyWeak<Info<T>>, Range<usize>)]> {
        let mut life_time = self.0.into_iter().collect::<Box<_>>();
        life_time.sort_by(|(_, l), (_, r)| {
            use std::cmp::Ordering::Equal;
            match l.start.cmp(&r.start) {
                Equal => r.end.cmp(&l.end),
                ord => ord,
            }
        });
        life_time
    }

    fn get_or_insert<'a>(&'a mut self, blob: &Rc<Info<T>>) -> Option<&'a mut Range<usize>> {
        match **blob {
            Info::Internal(_) => Some(
                self.0
                    .entry(KeyWeak::from(blob))
                    .or_insert_with(|| usize::MAX..0),
            ),
            _ => None,
        }
    }
}
