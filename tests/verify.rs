use jv::verify::{VerifyError, brute_force, certify};
use jv::{Assignment, DenseMatrix, Duals};

#[test]
fn brute_force_square() {
    let m = DenseMatrix::from_rows(&[[4, 1, 3], [2, 0, 5], [3, 2, 2]]).unwrap();
    assert_eq!(brute_force(&m), Some(5));
}

#[test]
fn brute_force_rectangular() {
    let m = DenseMatrix::from_rows(&[[10.0, 1.0, 7.0, 3.0], [2.0, 9.0, 8.0, 4.0]]).unwrap();
    assert_eq!(brute_force(&m), Some(3.0));
}

#[test]
fn shape_mismatch() {
    assert!(DenseMatrix::from_slice(&[1, 2, 3], 2, 2).is_err());
}

#[test]
fn certify_accepts_optimal_and_rejects_bad_duals() {
    let m = DenseMatrix::from_rows(&[[4, 1, 3], [2, 0, 5], [3, 2, 2]]).unwrap();
    // Optimal: (0,1)=1, (1,0)=2, (2,2)=2 => 5.
    let mut a = Assignment {
        row_to_col: vec![Some(1), Some(0), Some(2)],
        col_to_row: vec![Some(1), Some(0), Some(2)],
        cost: 5,
        duals: Some(Duals {
            u: vec![1, 2, 2],
            v: vec![0, 0, 0],
        }),
    };
    // Reduced cost (1,1) = 0 - 2 - 0 < 0, so these duals are infeasible.
    assert_eq!(
        certify(&m, &a, 0),
        Err(VerifyError::DualInfeasible { row: 1, col: 1 })
    );

    a.duals = Some(Duals {
        u: vec![1, 0, 1],
        v: vec![2, 0, 1],
    });
    assert_eq!(certify(&m, &a, 0), Ok(()));
}

#[test]
fn certify_rejects_inconsistent_reverse_mappings() {
    let costs = DenseMatrix::from_rows(&[[0, 0]]).unwrap();
    let mut result = jv::minimize(&costs).unwrap();
    let unassigned = result.col_to_row.iter().position(Option::is_none).unwrap();
    for row in [0, 1, usize::MAX] {
        result.col_to_row[unassigned] = Some(row);
        assert_eq!(
            certify(&costs, &result, 0),
            Err(VerifyError::Inconsistent {
                row,
                col: unassigned
            })
        );
    }
}

#[test]
fn certify_rejects_nonfinite_numbers_and_invalid_tolerances() {
    let costs = DenseMatrix::from_rows(&[[0.0]]).unwrap();
    let valid = jv::minimize(&costs).unwrap();
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut result = valid.clone();
        result.cost = invalid;
        assert!(certify(&costs, &result, 0.0).is_err());
        result = valid.clone();
        result.duals.as_mut().unwrap().u[0] = invalid;
        result.duals.as_mut().unwrap().v[0] = -invalid;
        assert!(certify(&costs, &result, 0.0).is_err());
        result = valid.clone();
        result.duals.as_mut().unwrap().v[0] = invalid;
        assert!(certify(&costs, &result, 0.0).is_err());
    }
    for eps in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        assert!(certify(&costs, &valid, eps).is_err());
    }
    let costs = DenseMatrix::from_rows(&[[1.0, 0.0]]).unwrap();
    let nonoptimal = Assignment {
        row_to_col: vec![Some(0)],
        col_to_row: vec![Some(0), None],
        cost: 1.0,
        duals: Some(Duals {
            u: vec![0.0],
            v: vec![0.0, 0.0],
        }),
    };
    assert!(certify(&costs, &nonoptimal, f64::NAN).is_err());
}

#[test]
fn certify_handles_malformed_integer_arithmetic_without_panicking() {
    let costs = DenseMatrix::from_rows(&[[0i64]]).unwrap();
    let mut result = jv::minimize(&costs).unwrap();
    result.cost = i64::MIN;
    assert!(certify(&costs, &result, 0).is_err());
    result.cost = 0;
    result.duals = Some(Duals {
        u: vec![i64::MIN],
        v: vec![i64::MAX],
    });
    assert!(certify(&costs, &result, 0).is_err());
}

#[test]
fn certify_rejects_invalid_matrix_values_and_shapes() {
    let costs = DenseMatrix::from_rows(&[[0.0, 0.0]]).unwrap();
    let result = jv::minimize(&costs).unwrap();
    for col in 0..2 {
        let mut values = [0.0, 0.0];
        values[col] = f64::NAN;
        let invalid = DenseMatrix::from_slice(&values, 1, 2).unwrap();
        assert!(certify(&invalid, &result, 0.0).is_err());
    }
    let tall = DenseMatrix::from_rows(&[[0], [0]]).unwrap();
    let result = jv::minimize(&tall).unwrap();
    assert!(certify(&tall, &result, 0).is_err());
}
