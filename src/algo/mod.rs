//! Assignment algorithms.

mod assertions;
mod jv;

pub use jv::JvSolver;

use crate::{Assignment, Control, Cost, CostMatrix, LapError};

/// Algorithm selection for the high-level API.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Algorithm {
    /// Pick based on shape, density and cost type.
    #[default]
    Auto,
    /// Jonker–Volgenant (dense).
    Jv,
}

/// A minimisation LAP algorithm.
///
/// Solvers accept any shape: with more rows than columns, some rows stay
/// unassigned. Maximisation is handled by the front end ([`crate::Lap`]).
/// Implementations own their work buffers, so reusing a solver for problems
/// of similar size does not reallocate them.
pub trait Solver<C: Cost> {
    fn solve<M, K>(&mut self, costs: &M, control: &mut K) -> Result<Assignment<C>, LapError>
    where
        M: CostMatrix<Cost = C>,
        K: Control;
}
