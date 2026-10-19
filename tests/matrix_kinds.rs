//! Structured and adversarial matrix families, beyond uniform random costs.

mod common;

use common::{Rng, brute_force_any, certify_any};
use jv::{Cost, DenseMatrix, LapError};

fn build<C: jv::Cost>(
    rows: usize,
    cols: usize,
    mut f: impl FnMut(usize, usize) -> C,
) -> DenseMatrix<'static, C> {
    let data = (0..rows * cols).map(|k| f(k / cols, k % cols)).collect();
    DenseMatrix::from_vec(data, rows, cols).unwrap()
}

const SHAPES: [(usize, usize); 6] = [
    (200, 200),
    (150, 300),
    (300, 150),
    (1, 200),
    (200, 1),
    (199, 200),
];

/// Families with many ties or strong structure, which stress the
/// initialisation and the equality checks in the path search.
fn int_families(
    rows: usize,
    cols: usize,
    rng: &mut Rng,
) -> Vec<(&'static str, DenseMatrix<'static, i64>)> {
    let (r, c) = (rows as i64, cols as i64);
    vec![
        ("constant", build(rows, cols, |_, _| 7)),
        (
            "i*j (Machol-Wien)",
            build(rows, cols, |i, j| (i * j) as i64),
        ),
        (
            "(i+1)*(j+1)",
            build(rows, cols, |i, j| ((i + 1) * (j + 1)) as i64),
        ),
        ("i+j", build(rows, cols, |i, j| (i + j) as i64)),
        (
            "|i-j|",
            build(rows, cols, |i, j| (i as i64 - j as i64).abs()),
        ),
        (
            "identical rows",
            build(rows, cols, |_, j| (j * 37 % 11) as i64),
        ),
        (
            "identical cols",
            build(rows, cols, |i, _| (i * 37 % 11) as i64),
        ),
        (
            "anti-diagonal zero",
            build(rows, cols, |i, j| {
                if i + j == rows.min(cols) - 1 { 0 } else { 100 }
            }),
        ),
        (
            "one cheap column",
            build(rows, cols, |_, j| if j == 0 { 0 } else { 1 }),
        ),
        ("binary", build(rows, cols, |_, _| rng.below(2) as i64)),
        (
            "mixed sign",
            build(rows, cols, |_, _| rng.below(2001) as i64 - 1000),
        ),
        (
            "wide range",
            build(rows, cols, |_, _| rng.below(1 << 40) as i64 - (1 << 39)),
        ),
        (
            "near validity bound",
            build(rows, cols, |_, _| {
                let b = i64::max_abs(rows.min(cols));
                if rng.below(2) == 0 {
                    b - rng.below(10) as i64
                } else {
                    -b + rng.below(10) as i64
                }
            }),
        ),
        ("geometric", {
            let pts: Vec<(i64, i64)> = (0..r.max(c))
                .map(|_| (rng.below(1000) as i64, rng.below(1000) as i64))
                .collect();
            let other: Vec<(i64, i64)> = (0..r.max(c))
                .map(|_| (rng.below(1000) as i64, rng.below(1000) as i64))
                .collect();
            build(rows, cols, |i, j| {
                (pts[i].0 - other[j].0).pow(2) + (pts[i].1 - other[j].1).pow(2)
            })
        }),
    ]
}

#[test]
fn structured_integer_families() {
    let mut rng = Rng::new(10);
    for (rows, cols) in SHAPES {
        for (name, m) in int_families(rows, cols, &mut rng) {
            let a = jv::minimize(&m).unwrap_or_else(|e| panic!("{name} {rows}x{cols}: {e}"));
            certify_any(&m, &a, 0);
            let b = jv::maximize(&m).unwrap_or_else(|e| panic!("{name} {rows}x{cols} max: {e}"));
            assert_eq!(b.cost, b.total_cost(&m), "{name}");
        }
    }
}

#[test]
fn structured_families_small_match_brute_force() {
    let mut rng = Rng::new(11);
    for rows in 1..=7 {
        for cols in 1..=7 {
            for (name, m) in int_families(rows, cols, &mut rng) {
                if name == "near validity bound" || name == "wide range" {
                    continue; // brute force sums would be fine, but keep it simple
                }
                let a = jv::minimize(&m).unwrap();
                assert_eq!(a.cost, brute_force_any(&m), "{name} {rows}x{cols}");
            }
        }
    }
}

#[test]
fn float_families() {
    let mut rng = Rng::new(12);
    for (rows, cols) in SHAPES {
        let families: Vec<(&str, DenseMatrix<'static, f64>, f64)> = vec![
            (
                "integers as floats",
                build(rows, cols, |_, _| rng.below(5) as f64),
                0.0,
            ),
            ("negative", build(rows, cols, |_, _| -rng.unit()), 1e-9),
            (
                "many magnitudes",
                build(rows, cols, |_, _| {
                    rng.unit() * 10f64.powi(rng.below(13) as i32 - 6)
                }),
                1e-3,
            ),
            (
                "tiny",
                build(rows, cols, |_, _| rng.unit() * 1e-300),
                1e-310,
            ),
            ("huge", build(rows, cols, |_, _| rng.unit() * 1e300), 1e288),
            (
                "near validity bound",
                build(rows, cols, |_, _| {
                    let b = f64::max_abs(rows.min(cols));
                    if rng.below(2) == 0 {
                        b * (1.0 - rng.unit() * 1e-3)
                    } else {
                        -b
                    }
                }),
                f64::max_abs(rows.min(cols)) * 1e-12,
            ),
            (
                "thirds",
                build(rows, cols, |i, j| ((i * 7 + j * 3) % 10) as f64 / 3.0),
                1e-9,
            ),
        ];
        for (name, m, eps) in families {
            let a = jv::minimize(&m).unwrap_or_else(|e| panic!("{name} {rows}x{cols}: {e}"));
            certify_any(&m, &a, eps);
        }
    }
}

#[test]
fn large_square_with_many_ties() {
    let mut rng = Rng::new(13);
    for n in [300, 1000] {
        for max in [0, 1, 3, 100] {
            let m = rng.int_matrix(n, n, max);
            let a = jv::minimize(&m).unwrap();
            certify_any(&m, &a, 0);
        }
    }
}

#[test]
fn i32_and_isize() {
    let mut rng = Rng::new(14);
    let m: DenseMatrix<i32> = build(100, 120, |_, _| rng.below(1 << 20) as i32 - (1 << 19));
    certify_any(&m, &jv::minimize(&m).unwrap(), 0);
    let m: DenseMatrix<isize> = build(120, 100, |_, _| rng.below(50) as isize);
    certify_any(&m, &jv::minimize(&m).unwrap(), 0);
}

#[test]
fn cost_bound_is_exact() {
    for (rows, cols) in [(3, 3), (40, 20), (20, 40)] {
        let b = i64::max_abs(rows.min(cols));
        let at = build(rows, cols, |i, j| if (i + j) % 2 == 0 { b } else { -b });
        certify_any(&at, &jv::minimize(&at).unwrap(), 0);
        jv::maximize(&at).unwrap();

        let above = build(rows, cols, |i, j| if (i, j) == (1, 2) { b + 1 } else { 0 });
        assert_eq!(
            jv::minimize(&above),
            Err(LapError::InvalidCost { row: 1, col: 2 })
        );
        let below = build(rows, cols, |i, j| if (i, j) == (2, 1) { -b - 1 } else { 0 });
        assert_eq!(
            jv::maximize(&below),
            Err(LapError::InvalidCost { row: 2, col: 1 })
        );
    }
}

#[test]
fn floating_point_bounds_keep_totals_finite() {
    let n = 200;
    for sign in [-1.0, 1.0] {
        let value = sign * f64::max_abs(n);
        let costs = DenseMatrix::from_vec(vec![value; n * n], n, n).unwrap();
        let result = jv::minimize(&costs).unwrap();
        assert!(result.cost.is_finite());
        certify_any(&costs, &result, value.abs() * 1e-12);
        assert!(jv::maximize(&costs).unwrap().cost.is_finite());

        let value = sign as f32 * f32::max_abs(n);
        let costs = DenseMatrix::from_vec(vec![value; n * n], n, n).unwrap();
        let result = jv::minimize(&costs).unwrap();
        assert!(result.cost.is_finite());
        certify_any(&costs, &result, value.abs() * 1e-5);
        assert!(jv::maximize(&costs).unwrap().cost.is_finite());
    }
}
