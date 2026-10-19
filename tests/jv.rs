mod common;

use std::ops::ControlFlow;

use common::{Rng, brute_force_any, certify_any};
use jv::matrix::{Negated, Transposed};
use jv::{CostMatrix, DenseMatrix, JvSolver, Lap, LapError, NoControl, Solver};

/// Every shape up to 7×7 with several value ranges (small ranges give many ties,
/// which exercise the degenerate paths of the initialisation).
#[test]
fn small_integer_instances_match_brute_force() {
    let mut rng = Rng::new(1);
    let mut solver = JvSolver::new();
    for rows in 1..=7 {
        for cols in 1..=7 {
            for &max in &[0, 1, 3, 10, 1000] {
                for _ in 0..20 {
                    let m = rng.int_matrix(rows, cols, max);
                    let a = solver.solve(&m, &mut NoControl).unwrap();
                    assert_eq!(
                        a.cost,
                        brute_force_any(&m),
                        "{rows}x{cols} max={max}\n{m:?}"
                    );
                    assert_eq!(a.cost, a.total_cost(&m));
                    assert_eq!(a.pairs().count(), rows.min(cols));
                    certify_any(&m, &a, 0);
                }
            }
        }
    }
}

#[test]
fn negative_integer_costs() {
    let mut rng = Rng::new(2);
    for n in 1..=7 {
        for _ in 0..50 {
            let data = (0..n * n).map(|_| rng.below(201) as i32 - 100).collect();
            let m = DenseMatrix::from_vec(data, n, n).unwrap();
            let a = jv::minimize(&m).unwrap();
            assert_eq!(a.cost, brute_force_any(&m));
            certify_any(&m, &a, 0);
        }
    }
}

#[test]
fn maximize_matches_brute_force_on_negated_costs() {
    let mut rng = Rng::new(3);
    for (rows, cols) in [(4, 4), (3, 6), (6, 3), (7, 7)] {
        for _ in 0..30 {
            let m = rng.int_matrix(rows, cols, 50);
            let a = jv::maximize(&m).unwrap();
            assert_eq!(a.cost, -brute_force_any(&Negated(&m)));
            assert_eq!(a.cost, a.total_cost(&m));
        }
    }
}

#[test]
fn large_float_instances_are_certified() {
    let mut rng = Rng::new(4);
    let mut solver = JvSolver::new();
    for &n in &[2, 10, 50, 200, 500] {
        for _ in 0..3 {
            let m = rng.float_matrix(n, n);
            let a = solver.solve(&m, &mut NoControl).unwrap();
            certify_any(&m, &a, 1e-9);
        }
    }
}

#[test]
fn large_rectangular_instances_are_certified() {
    let mut rng = Rng::new(5);
    for (rows, cols) in [(100, 300), (300, 100), (1, 500), (500, 1), (250, 251)] {
        let m = rng.float_matrix(rows, cols);
        let a = jv::minimize(&m).unwrap();
        certify_any(&m, &a, 1e-9);

        let m = rng.int_matrix(rows, cols, 20);
        let a = jv::minimize(&m).unwrap();
        certify_any(&m, &a, 0);
    }
}

#[test]
fn non_contiguous_views_give_same_result() {
    let mut rng = Rng::new(6);
    let m = rng.int_matrix(30, 45, 100);
    let a = jv::minimize(&m).unwrap();
    let t = jv::minimize(&Transposed(&m)).unwrap();
    assert_eq!(a.cost, t.cost);
    certify_any(&Transposed(&m), &t, 0);
}

#[test]
fn solver_reuse_across_sizes() {
    let mut rng = Rng::new(7);
    let mut solver = JvSolver::new();
    for (rows, cols) in [(50, 50), (3, 8), (8, 3), (1, 1), (0, 4), (60, 60)] {
        let m = rng.int_matrix(rows, cols, 30);
        let a = solver.solve(&m, &mut NoControl).unwrap();
        certify_any(&m, &a, 0);
    }
}

#[test]
fn empty_matrices() {
    for (rows, cols) in [(0, 0), (0, 3), (3, 0)] {
        let m = DenseMatrix::<f64>::from_vec(vec![], rows, cols).unwrap();
        let a = jv::minimize(&m).unwrap();
        assert_eq!(a.cost, 0.0);
        assert_eq!(a.row_to_col, vec![None; rows]);
        assert_eq!(a.col_to_row, vec![None; cols]);
    }
}

#[test]
fn invalid_costs_are_rejected() {
    let m = DenseMatrix::from_rows(&[[1.0, 2.0, 3.0], [4.0, f64::NAN, 6.0]]).unwrap();
    assert_eq!(
        jv::minimize(&m),
        Err(LapError::InvalidCost { row: 1, col: 1 })
    );

    // Tall matrix: the position must still refer to the original orientation.
    let m = DenseMatrix::from_rows(&[[1.0, 2.0], [3.0, 4.0], [f64::INFINITY, 6.0]]).unwrap();
    assert_eq!(
        jv::minimize(&m),
        Err(LapError::InvalidCost { row: 2, col: 0 })
    );

    let m = DenseMatrix::from_rows(&[[1, i64::MAX], [3, 4]]).unwrap();
    assert_eq!(
        jv::minimize(&m),
        Err(LapError::InvalidCost { row: 0, col: 1 })
    );
}

#[test]
fn maximize_rejects_integer_minima_before_negation() {
    for (rows, cols) in [(1, 1), (2, 3), (3, 2)] {
        let mut values = vec![0i64; rows * cols];
        values[rows * cols - 1] = i64::MIN;
        let costs = DenseMatrix::from_slice(&values, rows, cols).unwrap();
        assert_eq!(
            jv::maximize(&costs),
            Err(LapError::InvalidCost {
                row: rows - 1,
                col: cols - 1
            })
        );
    }
    let costs = DenseMatrix::from_rows(&[[i32::MIN]]).unwrap();
    assert_eq!(
        jv::maximize(&costs),
        Err(LapError::InvalidCost { row: 0, col: 0 })
    );
    let costs = DenseMatrix::from_rows(&[[isize::MIN]]).unwrap();
    assert_eq!(
        jv::maximize(&costs),
        Err(LapError::InvalidCost { row: 0, col: 0 })
    );
}

#[test]
fn cancellation() {
    let mut rng = Rng::new(8);
    let m = rng.float_matrix(20, 40);
    let mut ticks = 0;
    let mut stop_after_five = || {
        ticks += 1;
        if ticks > 5 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    };
    assert_eq!(
        Lap::new().solve_with(&m, &mut stop_after_five),
        Err(LapError::Cancelled)
    );

    // The solver stays usable after a cancelled run.
    let mut solver = JvSolver::new();
    let mut stop = || ControlFlow::Break(());
    assert_eq!(solver.solve(&m, &mut stop), Err(LapError::Cancelled));
    let a = solver.solve(&m, &mut NoControl).unwrap();
    certify_any(&m, &a, 1e-9);
}

#[test]
fn f32_costs() {
    let mut rng = Rng::new(9);
    let data: Vec<f32> = (0..64).map(|_| rng.below(100) as f32).collect();
    let m = DenseMatrix::from_slice(&data, 8, 8).unwrap();
    let a = jv::minimize(&m).unwrap();
    assert_eq!(a.cost, brute_force_any(&m));
    assert_eq!(m.rows(), 8);
}
