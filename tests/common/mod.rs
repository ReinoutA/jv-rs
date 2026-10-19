#![allow(dead_code)]

use jv::matrix::Transposed;
use jv::verify::{brute_force, certify};
use jv::{Assignment, Cost, CostMatrix, DenseMatrix, Duals};

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..bound`.
    pub fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }

    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn int_matrix(&mut self, rows: usize, cols: usize, max: i64) -> DenseMatrix<'static, i64> {
        let data = (0..rows * cols)
            .map(|_| self.below(max as u64 + 1) as i64)
            .collect();
        DenseMatrix::from_vec(data, rows, cols).unwrap()
    }

    pub fn float_matrix(&mut self, rows: usize, cols: usize) -> DenseMatrix<'static, f64> {
        let data = (0..rows * cols).map(|_| self.unit()).collect();
        DenseMatrix::from_vec(data, rows, cols).unwrap()
    }
}

/// Swap rows and columns of an assignment (including duals).
pub fn transpose<C: Clone>(a: &Assignment<C>) -> Assignment<C> {
    Assignment {
        row_to_col: a.col_to_row.clone(),
        col_to_row: a.row_to_col.clone(),
        cost: a.cost.clone(),
        duals: a.duals.as_ref().map(|d| Duals {
            u: d.v.clone(),
            v: d.u.clone(),
        }),
    }
}

/// Certify a minimisation result of any shape.
pub fn certify_any<M: CostMatrix>(m: &M, a: &Assignment<M::Cost>, eps: M::Cost) {
    let res = if m.rows() <= m.cols() {
        certify(m, a, eps)
    } else {
        certify(&Transposed(m), &transpose(a), eps)
    };
    if let Err(e) = res {
        panic!("certificate failed: {e:?}\n{a:?}");
    }
}

/// Brute-force optimum of a minimisation problem of any shape.
pub fn brute_force_any<M: CostMatrix>(m: &M) -> M::Cost
where
    M::Cost: Cost,
{
    if m.rows() <= m.cols() {
        brute_force(m)
    } else {
        brute_force(&Transposed(m))
    }
    .unwrap()
}
