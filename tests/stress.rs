//! Long randomized stress test. Run with overflow checks in release mode:
//!
//! ```text
//! CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS=true cargo test --release --test stress -- --ignored
//! ```

mod common;

use common::{Rng, brute_force_any, certify_any};
use jv::matrix::Negated;
use jv::{Assignment, Cost, DenseMatrix, Duals};

fn random_matrix(rng: &mut Rng, rows: usize, cols: usize) -> DenseMatrix<'static, i64> {
    let bound = i64::max_abs(rows.min(cols));
    let family = rng.below(6);
    let max = [0, 1, 2, 5, 100, 1 << 30][rng.below(6) as usize];
    let data = (0..rows * cols)
        .map(|k| {
            let (i, j) = ((k / cols) as i64, (k % cols) as i64);
            match family {
                0 => rng.below(max as u64 + 1) as i64,
                1 => rng.below(2 * max as u64 + 1) as i64 - max,
                2 => {
                    if rng.below(2) == 0 {
                        bound - rng.below(3) as i64
                    } else {
                        -bound + rng.below(3) as i64
                    }
                }
                3 => i * j % (max + 1),
                4 => (i - j).abs() * (rng.below(3) as i64),
                _ => {
                    if rng.below(10) == 0 {
                        -bound
                    } else {
                        rng.below(max as u64 + 1) as i64
                    }
                }
            }
        })
        .collect();
    DenseMatrix::from_vec(data, rows, cols).unwrap()
}

/// A maximisation result viewed as a minimisation result on the negated matrix.
fn as_min(a: &Assignment<i64>) -> Assignment<i64> {
    let neg = |v: &Vec<i64>| v.iter().map(|x| -x).collect();
    Assignment {
        row_to_col: a.row_to_col.clone(),
        col_to_row: a.col_to_row.clone(),
        cost: -a.cost,
        duals: a.duals.as_ref().map(|d| Duals {
            u: neg(&d.u),
            v: neg(&d.v),
        }),
    }
}

#[test]
#[ignore]
fn stress_random_instances() {
    let mut rng = Rng::new(0xC0FFEE);
    for iter in 0..200_000 {
        let small = iter % 4 == 0;
        let limit = if small { 7 } else { 80 };
        let (rows, cols) = (1 + rng.below(limit) as usize, 1 + rng.below(limit) as usize);
        let m = random_matrix(&mut rng, rows, cols);

        let a = jv::minimize(&m).unwrap();
        certify_any(&m, &a, 0);
        let b = jv::maximize(&m).unwrap();
        certify_any(&Negated(&m), &as_min(&b), 0);

        if small {
            assert_eq!(a.cost, brute_force_any(&m), "iter {iter}\n{m:?}");
            assert_eq!(b.cost, -brute_force_any(&Negated(&m)), "iter {iter}\n{m:?}");
        }
    }
}
