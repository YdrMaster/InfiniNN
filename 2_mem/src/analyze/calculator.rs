use std::{
    cmp::Ordering,
    collections::{BTreeSet, HashMap},
    ops::Range,
    usize,
};

#[derive(Debug)]
pub(crate) struct OffsetCalculator {
    alignment: usize,
    output_range: Range<usize>,
    free_list: BTreeSet<Area>,
    heads: HashMap<usize, usize>,
    tails: HashMap<usize, usize>,
}

impl OffsetCalculator {
    pub fn new(alignment: usize) -> Self {
        Self {
            alignment,
            output_range: usize::MAX..0,
            free_list: BTreeSet::new(),
            heads: HashMap::new(),
            tails: HashMap::new(),
        }
    }

    pub fn output_range(&self) -> Range<usize> {
        self.output_range.clone()
    }

    pub fn put(&mut self, range: &Range<usize>) {
        let len = range.len().div_ceil(self.alignment) * self.alignment;
        if len == 0 {
            return;
        }

        let mut head = range.start;
        let mut tail = head + len;
        if let Some(len_) = self.tails.remove(&head) {
            head -= len_;
            assert!(self.free_list.remove(&Area {
                off: head,
                len: len_,
            }));
            assert_eq!(self.heads.remove(&head), Some(len_));
        }
        if let Some(len_) = self.heads.remove(&tail) {
            assert!(self.free_list.remove(&Area {
                off: tail,
                len: len_,
            }));
            tail += len_;
            assert_eq!(self.tails.remove(&tail), Some(len_));
        }

        self.insert_area(Area {
            off: head,
            len: tail - head,
        })
    }

    pub fn take(&mut self, expect: usize) -> Option<Range<usize>> {
        let len = expect.div_ceil(self.alignment) * self.alignment;
        if len == 0 {
            return Some(usize::MAX..usize::MAX);
        }

        let &free = self.free_list.range(Area { off: 0, len }..).next()?;

        let head = free.off;
        let tail = free.off + free.len;

        self.free_list.remove(&free);
        self.heads.remove(&head);
        self.tails.remove(&tail);

        if free.len > len {
            self.insert_area(Area {
                off: free.off + len,
                len: free.len - len,
            })
        }

        let tail = head + expect;
        self.output_range.start = self.output_range.start.min(head);
        self.output_range.end = self.output_range.end.max(tail);
        Some(head..head + expect)
    }

    fn insert_area(&mut self, area: Area) {
        self.free_list.insert(area);
        self.heads.insert(area.off, area.len);
        self.tails.insert(area.off + area.len, area.len);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Area {
    off: usize,
    len: usize,
}

impl PartialOrd for Area {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Area {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.len.cmp(&other.len) {
            Ordering::Equal => self.off.cmp(&other.off),
            ord => ord,
        }
    }
}
