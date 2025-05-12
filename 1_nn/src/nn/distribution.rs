use mem::Tensor;
use std::{any::Any, hash::Hash, rc::Rc};

/// 分布式切分方式
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Distribution {
    pub start: usize,
    pub len: usize,
    pub total: usize,
}

impl Distribution {
    pub const MONO: Self = Self {
        start: 0,
        len: 1,
        total: 1,
    };

    pub fn new(start: usize, len: usize, total: usize) -> Self {
        assert!(0 < len && start + len <= total);
        Self { start, len, total }
    }

    #[inline]
    pub const fn is_mono(&self) -> bool {
        self.len == self.total
    }
}

#[derive(Clone)]
pub struct TPTensor<T> {
    pub act: Option<TPAction>,
    pub val: T,
}

impl<T> From<T> for TPTensor<T> {
    fn from(value: T) -> Self {
        Self {
            act: None,
            val: value,
        }
    }
}

#[derive(Clone)]
pub struct TPAction {
    pub wt: Rc<dyn WeightType>,
    pub dist: Distribution,
}

impl TPAction {
    pub fn new<WT: WeightType>(wt: WT, dist: Distribution) -> Self {
        Self {
            wt: Rc::new(wt),
            dist,
        }
    }
}

impl PartialEq for TPAction {
    fn eq(&self, other: &Self) -> bool {
        self.wt.check_eq(&*other.wt) && self.dist == other.dist
    }
}

impl Eq for TPAction {}

impl Hash for TPAction {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.wt.type_id().hash(state);
        self.dist.hash(state);
    }
}

pub trait WeightType: Any {
    fn move_data(&self, dist: Distribution, dst: &mut [u8], src: &Tensor<&[u8], 2>);
    fn check_eq(&self, other: &dyn Any) -> bool;
}

pub mod weight_types {
    use super::*;

    #[derive(Clone, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct AttnQKV(pub usize);

    #[derive(Clone, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct FfnGateUp;

    #[derive(Clone, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct ColumnTPWeight;

    #[derive(Clone, PartialEq, Eq)]
    #[repr(transparent)]
    pub struct RowTPWeight;

    macro_rules! impl_wt_eq {
        () => {
            fn check_eq(&self, other: &dyn Any) -> bool {
                match other.downcast_ref::<Self>() {
                    Some(other) => self.eq(other),
                    _ => false,
                }
            }
        };
    }

    impl WeightType for AttnQKV {
        impl_wt_eq!();
        fn move_data(&self, dist: Distribution, dst: &mut [u8], src: &Tensor<&[u8], 2>) {
            match src.layout().ndim() {
                1 => todo!(),
                2 => todo!(),
                _ => unreachable!(),
            }
        }
    }

    impl WeightType for FfnGateUp {
        impl_wt_eq!();
        fn move_data(&self, dist: Distribution, dst: &mut [u8], src: &Tensor<&[u8], 2>) {
            match src.layout().ndim() {
                1 => todo!(),
                2 => todo!(),
                _ => unreachable!(),
            }
        }
    }

    impl WeightType for ColumnTPWeight {
        impl_wt_eq!();
        fn move_data(&self, dist: Distribution, dst: &mut [u8], src: &Tensor<&[u8], 2>) {
            match src.layout().ndim() {
                1 => todo!(),
                2 => todo!(),
                _ => unreachable!(),
            }
        }
    }

    impl WeightType for RowTPWeight {
        impl_wt_eq!();
        fn move_data(&self, dist: Distribution, dst: &mut [u8], src: &Tensor<&[u8], 2>) {
            match src.layout().ndim() {
                1 => todo!(),
                2 => todo!(),
                _ => unreachable!(),
            }
        }
    }
}
