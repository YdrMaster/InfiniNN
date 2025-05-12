use super::key_weak::KeyWeak;
use crate::{Graph, Info};
use graph::NodeRef;
use std::{cmp::Ordering, collections::HashMap, iter::zip, ops::Range, rc::Rc};

pub struct BlobLifeTime<T> {
    pub blob: KeyWeak<Info<T>>,
    pub life_time: Range<usize>,
}

impl<T> PartialEq for BlobLifeTime<T> {
    fn eq(&self, other: &Self) -> bool {
        self.blob == other.blob
    }
}
impl<T> Eq for BlobLifeTime<T> {}

impl<T> PartialOrd for BlobLifeTime<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T> Ord for BlobLifeTime<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.life_time.start.cmp(&other.life_time.start) {
            Ordering::Equal => match other.life_time.end.cmp(&self.life_time.end) {
                Ordering::Equal => self.blob.cmp(&other.blob),
                ord => ord,
            },
            ord => ord,
        }
    }
}

impl<T> Graph<T> {
    pub fn blob_lifetime(&self) -> Box<[BlobLifeTime<T>]> {
        let Self(graph::Graph { topo, nodes, edges }) = self;
        // 在分析器中标记全图输入输出
        let mut analyzer = BlobAnalyzer::new(
            topo.n_node(),
            topo.global_inputs().map(|i| edges[i].get()),
            topo.global_outputs().iter().map(|&i| edges[i].get()),
        );
        // 根据算子生成每个 Blob 的生命周期
        for (i, (topo, node)) in zip(topo.iter(), nodes).enumerate() {
            if node.value.name == "empty" {
                continue;
            }
            let NodeRef { inputs, outputs } = topo;
            for &input in inputs {
                analyzer.push(i, edges[input].get())
            }
            for output in outputs {
                analyzer.push(i, edges[output].get())
            }
        }
        // 对生命周期排序
        analyzer.take()
    }
}

#[repr(transparent)]
struct BlobAnalyzer<T>(HashMap<KeyWeak<Info<T>>, Range<usize>>);

impl<T> BlobAnalyzer<T> {
    fn new<'a>(
        n_node: usize,
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
                record.end = n_node
            }
        }
        ans
    }

    fn push(&mut self, i_node: usize, blob: &Rc<Info<T>>) {
        if let Some(record) = self.get_or_insert(blob) {
            record.start = record.start.min(i_node);
            record.end = record.end.max(i_node);
        }
    }

    fn take(self) -> Box<[BlobLifeTime<T>]> {
        self.0
            .into_iter()
            .map(|(blob, life_time)| BlobLifeTime { blob, life_time })
            .collect::<Box<_>>()
    }

    fn get_or_insert<'a>(&'a mut self, blob: &Rc<Info<T>>) -> Option<&'a mut Range<usize>> {
        match **blob {
            Info::Internal(_) => Some(
                self.0
                    .entry(KeyWeak::from(blob))
                    .or_insert_with(|| super::EMPTY_RANGE),
            ),
            _ => None,
        }
    }
}
