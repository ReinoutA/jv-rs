use crate::{Cost, CostMatrix};

/// Dual variables: `u[i] + v[j] <= c[i][j]` for all pairs, with equality on
/// assigned pairs. Together they prove that an assignment is optimal.
#[derive(Debug, Clone, PartialEq)]
pub struct Duals<C> {
    pub u: Vec<C>,
    pub v: Vec<C>,
}

/// Result of solving an assignment problem.
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment<C> {
    /// Column assigned to each row (`None` only when rows > cols).
    pub row_to_col: Vec<Option<usize>>,
    /// Row assigned to each column (`None` for unused columns).
    pub col_to_row: Vec<Option<usize>>,
    /// Total cost in the original objective (not negated for maximisation).
    pub cost: C,
    /// Optimality certificate, if the solver provides one.
    pub duals: Option<Duals<C>>,
}

impl<C: Cost> Assignment<C> {
    /// Iterate over assigned `(row, col)` pairs.
    pub fn pairs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.row_to_col
            .iter()
            .enumerate()
            .filter_map(|(r, c)| c.map(|c| (r, c)))
    }

    /// Recompute the total cost against a matrix.
    pub fn total_cost<M: CostMatrix<Cost = C>>(&self, m: &M) -> C {
        self.pairs().fold(C::ZERO, |acc, (r, c)| acc + m.get(r, c))
    }
}
