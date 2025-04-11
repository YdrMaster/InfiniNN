use super::Tensor;

impl<T, const N: usize> Tensor<T, N> {
    pub fn index(self, indices: &[usize]) -> Self {
        let mut layout = self.layout.clone();
        for &index in indices {
            layout = layout.index(0, index)
        }
        Self {
            dt: self.dt,
            layout,
            item: self.item,
        }
    }

    pub fn merge(self, axis: usize, len: usize) -> Self {
        Self {
            dt: self.dt,
            layout: self.layout.merge_be(axis, len).unwrap(),
            item: self.item,
        }
    }
}
