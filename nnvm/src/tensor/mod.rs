mod nn;
mod saved;

use rw_rc::{LocalMut, LocalRef, RwRc};
use tensor::digit_layout::DigitLayout;

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
}

impl<T> Tensor<RwRc<T>> {
    /// 锁定张量，禁止再写。
    pub fn lock(&self) -> bool {
        // 锁定一个张量意味着尝试将所有子张量的数据置于读取状态
        // 如果锁定失败，应该恢复所有数据的状态
        match self {
            Self::Simple(tensor) => tensor.get().try_read_global().is_some(),
            Self::Grouped(tensors) => {
                if tensors.iter().all(|t| t.get().is_readable()) {
                    for t in tensors {
                        t.get().read_global();
                    }
                    true
                } else {
                    false
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
