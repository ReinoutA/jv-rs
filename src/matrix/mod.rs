mod dense;
mod transform;

pub use dense::DenseMatrix;
pub use transform::{Negated, Transposed};

use crate::Cost;

/// Read access to a `rows × cols` cost matrix.
pub trait CostMatrix {
    type Cost: Cost;

    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn get(&self, row: usize, col: usize) -> Self::Cost;

    /// The whole matrix as one contiguous row-major slice, if stored that way.
    /// Solvers use it directly; other matrices are copied into a work buffer.
    fn as_row_major(&self) -> Option<&[Self::Cost]> {
        None
    }
}

impl<M: CostMatrix + ?Sized> CostMatrix for &M {
    type Cost = M::Cost;

    fn rows(&self) -> usize {
        (**self).rows()
    }
    fn cols(&self) -> usize {
        (**self).cols()
    }
    #[inline]
    fn get(&self, row: usize, col: usize) -> Self::Cost {
        (**self).get(row, col)
    }
    fn as_row_major(&self) -> Option<&[Self::Cost]> {
        (**self).as_row_major()
    }
}
