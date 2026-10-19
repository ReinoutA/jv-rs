//! Correctness checks, intended for tests and debugging.

use crate::{Assignment, Cost, CostMatrix};

/// Optimal minimisation cost by trying all assignments. Only practical for
/// small inputs (about `rows <= 9`). Returns `None` if `rows > cols`.
pub fn brute_force<M: CostMatrix>(m: &M) -> Option<M::Cost> {
    if m.rows() > m.cols() {
        return None;
    }
    let mut used = vec![false; m.cols()];
    let mut best = None;
    search(m, 0, M::Cost::ZERO, &mut used, &mut best);
    best
}

fn search<M: CostMatrix>(
    m: &M,
    row: usize,
    acc: M::Cost,
    used: &mut [bool],
    best: &mut Option<M::Cost>,
) {
    if row == m.rows() {
        if best.is_none_or(|b| acc < b) {
            *best = Some(acc);
        }
        return;
    }
    for col in 0..m.cols() {
        if !used[col] {
            used[col] = true;
            search(m, row + 1, acc + m.get(row, col), used, best);
            used[col] = false;
        }
    }
}

/// Reasons an assignment fails [`certify`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum VerifyError {
    WrongShape,
    InvalidTolerance,
    WrongLength,
    NotComplete,
    Inconsistent { row: usize, col: usize },
    CostMismatch,
    MissingDuals,
    InvalidCost { row: usize, col: usize },
    InvalidRowDual { row: usize },
    InvalidColumnDual { col: usize },
    ArithmeticOverflow,
    DualInfeasible { row: usize, col: usize },
    NotTight { row: usize, col: usize },
    PositiveColumnPrice { col: usize },
    UnassignedColumnPrice { col: usize },
}

/// Check structure and, via the duals, optimality of a minimisation result
/// for a matrix with `rows <= cols`. Use `eps = 0` for integer costs.
/// The tolerance must be finite and nonnegative. Costs must satisfy
/// [`Cost::is_valid`]; totals and duals must be finite. Unrepresentable
/// intermediate arithmetic returns [`VerifyError::ArithmeticOverflow`].
///
/// For rectangular problems (`rows < cols`) the duals must also satisfy
/// `v[j] <= 0` for all columns and `v[j] = 0` for unassigned columns; otherwise
/// the certificate would not exclude a cheaper solution using other columns.
pub fn certify<M: CostMatrix>(
    m: &M,
    a: &Assignment<M::Cost>,
    eps: M::Cost,
) -> Result<(), VerifyError> {
    use VerifyError::*;
    let zero = M::Cost::ZERO;
    let (n, k) = (m.rows(), m.cols());
    if n > k {
        return Err(WrongShape);
    }
    if !eps.is_finite() || eps < zero {
        return Err(InvalidTolerance);
    }
    if a.row_to_col.len() != n || a.col_to_row.len() != k {
        return Err(WrongLength);
    }
    for (row, col) in a.row_to_col.iter().enumerate() {
        let col = col.ok_or(NotComplete)?;
        if a.col_to_row.get(col).copied().flatten() != Some(row) {
            return Err(Inconsistent { row, col });
        }
    }
    for (col, row) in a.col_to_row.iter().enumerate() {
        if let Some(row) = *row {
            if a.row_to_col.get(row).copied().flatten() != Some(col) {
                return Err(Inconsistent { row, col });
            }
        }
    }
    if !a.cost.is_finite() {
        return Err(CostMismatch);
    }
    let total = a.pairs().try_fold(zero, |total, (row, col)| {
        let cost = m.get(row, col);
        if !cost.is_valid(n) {
            return Err(InvalidCost { row, col });
        }
        total.checked_add(cost).ok_or(ArithmeticOverflow)
    })?;
    let difference = if total >= a.cost {
        total.checked_sub(a.cost)
    } else {
        a.cost.checked_sub(total)
    };
    if !difference.is_some_and(|difference| difference <= eps) {
        return Err(CostMismatch);
    }
    let d = a.duals.as_ref().ok_or(MissingDuals)?;
    if d.u.len() != n || d.v.len() != k {
        return Err(WrongLength);
    }
    for (row, &price) in d.u.iter().enumerate() {
        if !price.is_finite() {
            return Err(InvalidRowDual { row });
        }
    }
    for (col, &price) in d.v.iter().enumerate() {
        if !price.is_finite() {
            return Err(InvalidColumnDual { col });
        }
    }
    for row in 0..n {
        for col in 0..k {
            let cost = m.get(row, col);
            if !cost.is_valid(n) {
                return Err(InvalidCost { row, col });
            }
            let reduced = cost
                .checked_sub(d.u[row])
                .and_then(|value| value.checked_sub(d.v[col]))
                .ok_or(ArithmeticOverflow)?;
            if reduced < zero - eps {
                return Err(DualInfeasible { row, col });
            }
            if a.row_to_col[row] == Some(col) && reduced > eps {
                return Err(NotTight { row, col });
            }
        }
    }
    if n < k {
        for col in 0..k {
            if d.v[col] > eps {
                return Err(PositiveColumnPrice { col });
            }
            if a.col_to_row[col].is_none() && d.v[col] < zero - eps {
                return Err(UnassignedColumnPrice { col });
            }
        }
    }
    Ok(())
}
