//! Jonker–Volgenant algorithm for dense matrices.
//!
//! R. Jonker, A. Volgenant. A Shortest Augmenting Path Algorithm for Dense and
//! Sparse Linear Assignment Problems. Computing 38, 325–340 (1987).
//!
//! The solver works on a normalised `n × m` problem with `n <= m`; tall inputs
//! are transposed into the work buffer. Square problems use the full JV
//! initialisation (column reduction, reduction transfer, two rounds of
//! augmenting row reduction). Rectangular problems start from `v = 0` and use
//! only the shortest augmenting path phase: the column prices then only ever
//! decrease, so unassigned columns keep price 0 and the duals remain a valid
//! optimality certificate.
//!
//! Row duals are implicit throughout: for an assigned row `i`,
//! `u[i] = c[i][x[i]] - v[x[i]]`, and `x[i]` minimises `c[i][j] - v[j]`.

use crate::{Assignment, Control, Cost, CostMatrix, Duals, LapError, Solver};

const NONE: usize = usize::MAX;

/// Dense Jonker–Volgenant solver with reusable work buffers.
#[derive(Debug, Clone)]
pub struct JvSolver<C> {
    /// Copy of the costs when the input is not contiguous row-major or is transposed.
    scratch: Vec<C>,
    /// Column prices.
    v: Vec<C>,
    /// Column assigned to each row.
    x: Vec<usize>,
    /// Row assigned to each column.
    y: Vec<usize>,
    /// Shortest path distances.
    d: Vec<C>,
    /// Predecessor row of each column in the shortest path tree.
    pred: Vec<usize>,
    /// Column order: `[ready | scan | todo]` during a path search.
    cols: Vec<usize>,
    /// Free rows; also used as a flag array during column reduction.
    free: Vec<usize>,
}

impl<C: Cost> Default for JvSolver<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Cost> JvSolver<C> {
    pub fn new() -> Self {
        Self {
            scratch: Vec::new(),
            v: Vec::new(),
            x: Vec::new(),
            y: Vec::new(),
            d: Vec::new(),
            pred: Vec::new(),
            cols: Vec::new(),
            free: Vec::new(),
        }
    }

    fn reset(&mut self, n: usize, m: usize) {
        self.v.clear();
        self.v.resize(m, C::ZERO);
        self.x.clear();
        self.x.resize(n, NONE);
        self.y.clear();
        self.y.resize(m, NONE);
        self.d.clear();
        self.d.resize(m, C::ZERO);
        self.pred.clear();
        self.pred.resize(m, NONE);
        self.cols.clear();
        self.cols.resize(m, 0);
        self.free.clear();
    }

    /// Solve the normalised problem `c` (`n × m`, row-major, `n <= m`).
    fn run<K: Control>(
        &mut self,
        c: &[C],
        n: usize,
        m: usize,
        control: &mut K,
    ) -> Result<(), LapError> {
        self.reset(n, m);
        if n == 0 {
            return Ok(());
        }
        if n == m && n > 1 {
            self.column_reduction(c, n);
            for _ in 0..2 {
                if self.free.is_empty() {
                    break;
                }
                self.augmenting_row_reduction(c, n);
            }
        } else {
            self.free.extend(0..n);
        }

        let free = std::mem::take(&mut self.free);
        let mut result = Ok(());
        for &f in &free {
            if control.tick().is_break() {
                result = Err(LapError::Cancelled);
                break;
            }
            self.augment(c, m, f);
        }
        self.free = free;
        result
    }

    /// Column reduction with reduction transfer (square problems only, `n >= 2`).
    /// Fills `free` with the rows left unassigned.
    fn column_reduction(&mut self, c: &[C], n: usize) {
        let Self { v, x, y, free, .. } = self;

        // v[j] = min_i c[i][j], y[j] = argmin (first row wins ties).
        v.copy_from_slice(&c[..n]);
        y.fill(0);
        for i in 1..n {
            let row = &c[i * n..(i + 1) * n];
            for j in 0..n {
                if row[j] < v[j] {
                    v[j] = row[j];
                    y[j] = i;
                }
            }
        }

        // Assign each row to at most one of the columns it minimises.
        // `free` doubles as a flag: 1 if the row minimises exactly one column.
        free.clear();
        free.resize(n, 1);
        for j in (0..n).rev() {
            let i = y[j];
            if x[i] == NONE {
                x[i] = j;
            } else {
                free[i] = 0;
                y[j] = NONE;
            }
        }

        // Reduction transfer: lower the price of a uniquely assigned column as
        // far as possible while it stays row i's cheapest column.
        let mut n_free = 0;
        for i in 0..n {
            let j = x[i];
            if j == NONE {
                free[n_free] = i;
                n_free += 1;
            } else if free[i] == 1 {
                let row = &c[i * n..(i + 1) * n];
                let mut min = C::MAX;
                for k in 0..n {
                    if k != j {
                        let r = row[k] - v[k];
                        if r < min {
                            min = r;
                        }
                    }
                }
                v[j] = v[j] - min;
            }
        }
        free.truncate(n_free);
    }

    /// Augmenting row reduction (square problems only, `n >= 2`).
    /// Replaces `free` with the rows still unassigned afterwards.
    fn augmenting_row_reduction(&mut self, c: &[C], n: usize) {
        let Self { v, x, y, free, .. } = self;
        let n_free = free.len();
        let mut current = 0;
        let mut new_free = 0;
        let mut rr_cnt = 0usize;

        while current < n_free {
            rr_cnt += 1;
            let fi = free[current];
            current += 1;

            // Cheapest and second cheapest reduced cost in row fi.
            let row = &c[fi * n..(fi + 1) * n];
            let (mut j1, mut v1) = (0, row[0] - v[0]);
            let (mut j2, mut v2) = (NONE, C::MAX);
            debug_assert_eq!(row.len(), v.len());
            for (block, (costs, prices)) in row[1..].chunks(4).zip(v[1..n].chunks(4)).enumerate() {
                // Reject a whole block without serial updates to the minima.
                // Evaluate all four comparisons so the rejection can vectorize.
                if costs.len() == 4
                    && ((costs[0] - prices[0] >= v2)
                        & (costs[1] - prices[1] >= v2)
                        & (costs[2] - prices[2] >= v2)
                        & (costs[3] - prices[3] >= v2))
                {
                    continue;
                }
                for (offset, (&cost, &price)) in costs.iter().zip(prices).enumerate() {
                    let j = 1 + block * 4 + offset;
                    let r = cost - price;
                    if r < v2 {
                        if r >= v1 {
                            v2 = r;
                            j2 = j;
                        } else {
                            v2 = v1;
                            v1 = r;
                            j2 = j1;
                            j1 = j;
                        }
                    }
                }
            }
            debug_assert!(super::assertions::row_minima(row, v, (j1, v1), (j2, v2)));

            let mut i0 = y[j1];
            let v1_new = v[j1] - (v2 - v1);
            let v1_lowers = v1_new < v[j1];
            if rr_cnt < current * n {
                if v1_lowers {
                    v[j1] = v1_new;
                } else if i0 != NONE && j2 != NONE {
                    j1 = j2;
                    i0 = y[j2];
                }
                if i0 != NONE {
                    if v1_lowers {
                        // Process the displaced row next.
                        current -= 1;
                        free[current] = i0;
                    } else {
                        free[new_free] = i0;
                        new_free += 1;
                    }
                }
            } else if i0 != NONE {
                free[new_free] = i0;
                new_free += 1;
            }
            x[fi] = j1;
            y[j1] = fi;
        }
        free.truncate(new_free);
    }

    /// Find a shortest augmenting path from free row `f` (Dijkstra on reduced
    /// costs), update the column prices and augment along the path.
    fn augment(&mut self, c: &[C], m: usize, f: usize) {
        let Self {
            v,
            x,
            y,
            d,
            pred,
            cols,
            ..
        } = self;

        let row = &c[f * m..(f + 1) * m];
        for j in 0..m {
            cols[j] = j;
            pred[j] = f;
            d[j] = row[j] - v[j];
        }

        // cols[..lo]: ready (distance final and scanned),
        // cols[lo..hi]: distance == mu, waiting to be scanned,
        // cols[hi..]: todo.
        let (mut lo, mut hi) = (0, 0);
        let mut mu = C::ZERO;
        let sink = 'search: loop {
            if lo == hi {
                // Move all todo columns with minimal distance to the scan list.
                hi = lo + 1;
                mu = d[cols[lo]];
                let todo = hi;
                for k in todo..m {
                    let j = cols[k];
                    let dj = d[j];
                    if dj <= mu {
                        if dj < mu {
                            hi = lo;
                            mu = dj;
                        }
                        cols.swap(k, hi);
                        hi += 1;
                    }
                }
                for &j in &cols[lo..hi] {
                    if y[j] == NONE {
                        break 'search j;
                    }
                }
            }

            // Scan one column: relax the edges of the row assigned to it.
            let j = cols[lo];
            lo += 1;
            let i = y[j];
            let ri = &c[i * m..(i + 1) * m];
            let h = ri[j] - v[j] - mu;
            // Columns swapped to `hi` below were already examined in this loop.
            let todo = hi;
            for k in todo..m {
                let col = cols[k];
                let red = ri[col] - v[col] - h;
                if red < d[col] {
                    d[col] = red;
                    pred[col] = i;
                    if red == mu {
                        if y[col] == NONE {
                            break 'search col;
                        }
                        cols.swap(k, hi);
                        hi += 1;
                    }
                }
            }
        };

        // Keep reduced costs non-negative and the new path tight.
        for &j in &cols[..lo] {
            v[j] = v[j] + (d[j] - mu);
        }

        let mut j = sink;
        loop {
            let i = pred[j];
            y[j] = i;
            let next = x[i];
            x[i] = j;
            if i == f {
                break;
            }
            j = next;
        }
    }
}

impl<C: Cost> Solver<C> for JvSolver<C> {
    fn solve<M, K>(&mut self, costs: &M, control: &mut K) -> Result<Assignment<C>, LapError>
    where
        M: CostMatrix<Cost = C>,
        K: Control,
    {
        let (rows, cols) = (costs.rows(), costs.cols());
        let transposed = rows > cols;
        let (n, m) = if transposed {
            (cols, rows)
        } else {
            (rows, cols)
        };

        let mut scratch = std::mem::take(&mut self.scratch);
        let c: &[C] = match costs.as_row_major() {
            Some(data) if !transposed => data,
            _ => {
                scratch.clear();
                scratch.reserve(n * m);
                for i in 0..n {
                    for j in 0..m {
                        scratch.push(if transposed {
                            costs.get(j, i)
                        } else {
                            costs.get(i, j)
                        });
                    }
                }
                &scratch
            }
        };

        let result = validate(c, n, m, transposed)
            .and_then(|()| self.run(c, n, m, control))
            .map(|()| self.assignment(c, m, transposed));
        self.scratch = scratch;
        result
    }
}

impl<C: Cost> JvSolver<C> {
    /// Build the result from a completed run, in the original orientation.
    fn assignment(&self, c: &[C], m: usize, transposed: bool) -> Assignment<C> {
        let cost = self
            .x
            .iter()
            .enumerate()
            .fold(C::ZERO, |acc, (i, &j)| acc + c[i * m + j]);
        let u = self
            .x
            .iter()
            .enumerate()
            .map(|(i, &j)| c[i * m + j] - self.v[j])
            .collect();
        let mut duals = Duals {
            u,
            v: self.v.clone(),
        };

        let to_opt = |s: &[usize]| {
            s.iter()
                .map(|&k| (k != NONE).then_some(k))
                .collect::<Vec<_>>()
        };
        let (mut row_to_col, mut col_to_row) = (to_opt(&self.x), to_opt(&self.y));
        if transposed {
            std::mem::swap(&mut row_to_col, &mut col_to_row);
            std::mem::swap(&mut duals.u, &mut duals.v);
        }
        Assignment {
            row_to_col,
            col_to_row,
            cost,
            duals: Some(duals),
        }
    }
}

/// Reject unusable costs (see [`Cost::max_abs`]), reporting the position in the
/// original orientation. `n = min(rows, cols)` is the number of pairs.
fn validate<C: Cost>(c: &[C], n: usize, m: usize, transposed: bool) -> Result<(), LapError> {
    match c.iter().position(|v| !v.is_valid(n)) {
        None => Ok(()),
        Some(idx) => {
            let (i, j) = (idx / m, idx % m);
            let (row, col) = if transposed { (j, i) } else { (i, j) };
            Err(LapError::InvalidCost { row, col })
        }
    }
}
