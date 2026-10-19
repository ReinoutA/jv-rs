//! Zero-copy views used to normalise a problem to "minimise, rows <= cols".

use super::CostMatrix;
use crate::Cost;

/// Swaps rows and columns.
#[derive(Debug, Clone, Copy)]
pub struct Transposed<M>(pub M);

impl<M: CostMatrix> CostMatrix for Transposed<M> {
    type Cost = M::Cost;

    fn rows(&self) -> usize {
        self.0.cols()
    }
    fn cols(&self) -> usize {
        self.0.rows()
    }
    #[inline]
    fn get(&self, row: usize, col: usize) -> M::Cost {
        self.0.get(col, row)
    }
}

/// Negates every cost, turning maximisation into minimisation.
#[derive(Debug, Clone, Copy)]
pub struct Negated<M>(pub M);

impl<M: CostMatrix> CostMatrix for Negated<M> {
    type Cost = M::Cost;

    fn rows(&self) -> usize {
        self.0.rows()
    }
    fn cols(&self) -> usize {
        self.0.cols()
    }
    #[inline]
    fn get(&self, row: usize, col: usize) -> M::Cost {
        self.0.get(row, col).neg()
    }
}
