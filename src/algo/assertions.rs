use crate::Cost;

/// Recompute each minimum independently, preserving the first column on ties.
pub(super) fn row_minima<C: Cost>(
    row: &[C],
    prices: &[C],
    first: (usize, C),
    second: (usize, C),
) -> bool {
    assert_eq!(row.len(), prices.len());
    let minimum_except = |excluded| {
        row.iter()
            .zip(prices)
            .enumerate()
            .filter(|&(j, _)| j != excluded)
            .map(|(j, (&cost, &price))| (j, cost - price))
            .reduce(|best, candidate| {
                if candidate.1 < best.1 {
                    candidate
                } else {
                    best
                }
            })
    };
    minimum_except(usize::MAX) == Some(first) && minimum_except(first.0) == Some(second)
}
