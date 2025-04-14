mod backward;
mod kernel;
mod nn;

use rwrc::{LocalMut, LocalRef, RwRc};
use tensor::digit_layout::DigitLayout;

pub use backward::BackwardTensorOf;
pub use kernel::KernelTensorOf;
pub use nn::{NNTensor, NNTensorId};

#[derive(Clone)]
pub enum Tensor<T> {
    Simple(tensor::Tensor<T, 4>),
    Grouped(Box<[tensor::Tensor<T, 4>]>),
}

impl<T> Tensor<T> {
    pub fn dt(&self) -> DigitLayout {
        match self {
            Self::Simple(tensor) => tensor.dt(),
            Self::Grouped(tensors) => tensors[0].dt(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Simple(tensor) => tensor.shape(),
            Self::Grouped(tensors) => tensors[0].shape(),
        }
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Tensor<U> {
        match self {
            Self::Simple(tensor) => Tensor::Simple(tensor.map(&mut f)),
            Self::Grouped(tensors) => Tensor::Grouped(
                tensors
                    .into_iter()
                    .map(|t| t.map(&mut f))
                    .collect::<Box<_>>(),
            ),
        }
    }
}

impl<T> Tensor<RwRc<T>> {
    /// 锁定张量，禁止再写。
    pub fn lock(&mut self) -> bool {
        // 锁定一个张量意味着尝试将所有子张量的数据置于读取状态
        // 如果锁定失败，应该恢复所有数据的状态
        match self {
            Self::Simple(tensor) => tensor.get_mut().try_read_global(),
            Self::Grouped(tensors) => {
                if tensors.iter().all(|t| t.get().is_readable()) {
                    for t in tensors {
                        assert!(t.get_mut().try_read_global())
                    }
                    true
                } else {
                    false
                }
            }
        }
    }

    /// 解锁张量，不再阻止对张量写入。
    pub fn unlock(&mut self) {
        match self {
            Self::Simple(tensor) => tensor.get_mut().release(),
            Self::Grouped(tensors) => {
                if tensors.iter().all(|t| t.get().is_readable()) {
                    for t in tensors {
                        t.get_mut().release()
                    }
                } else {
                }
            }
        }
    }

    /// 判断张量是否可写。
    pub fn is_mutable(&self) -> bool {
        match self {
            Self::Simple(tensor) => tensor.get().is_writeable(),
            Self::Grouped(tensors) => tensors.iter().all(|t| t.get().is_writeable()),
        }
    }

    /// 生成张量的一个可读的副本。
    pub fn read(&self) -> Tensor<LocalRef<T>> {
        match self {
            Self::Simple(tensor) => Tensor::Simple(tensor.as_ref().map(RwRc::read)),
            Self::Grouped(tensors) => {
                Tensor::Grouped(tensors.iter().map(|t| t.as_ref().map(RwRc::read)).collect())
            }
        }
    }

    /// 生成张量的一个可写的副本。
    pub fn write(&mut self) -> Tensor<LocalMut<T>> {
        match self {
            Self::Simple(tensor) => Tensor::Simple(tensor.as_mut().map(RwRc::write)),
            Self::Grouped(tensors) => Tensor::Grouped(
                tensors
                    .iter_mut()
                    .map(|t| t.as_mut().map(RwRc::write))
                    .collect(),
            ),
        }
    }
}
