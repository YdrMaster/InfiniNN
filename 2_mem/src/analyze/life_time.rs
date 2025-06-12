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
                analyzer.record(edges[input].get(), i)
            }
            for output in outputs {
                analyzer.record(edges[output].get(), i)
            }
        }
        // 对生命周期排序
        analyzer.take()
    }
}

/// 块分析器
///
/// 一次遍历，记录所有块的生命周期区间。
#[repr(transparent)]
struct BlobAnalyzer<T>(HashMap<KeyWeak<Info<T>>, Range<usize>>);

impl<T> BlobAnalyzer<T> {
    /// 初始化分析器，标记全图输入输出块的生命周期。
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
            ans.record(blob, 0)
        }
        for blob in outputs {
            ans.record(blob, n_node)
        }
        ans
    }

    /// 标记 `blob` 在第 `i_node` 号节点处仍存在。
    fn record(&mut self, blob: &Rc<Info<T>>, i_node: usize) {
        if let Info::Internal(_) = **blob {
            use std::collections::hash_map::Entry::{Occupied, Vacant};
            match self.0.entry(KeyWeak::from(blob)) {
                Occupied(mut entry) => {
                    let record = entry.get_mut();
                    *record = record.start.min(i_node)..record.end.max(i_node)
                }
                Vacant(entry) => {
                    entry.insert(i_node..i_node);
                }
            }
        }
    }

    /// 提取分析结果。
    fn take(self) -> Box<[BlobLifeTime<T>]> {
        self.0
            .into_iter()
            .map(|(blob, life_time)| BlobLifeTime { blob, life_time })
            .collect()
    }
}
