# jv-rs

A dependency-free Rust library for dense linear assignment problems, based on
the Jonker–Volgenant algorithm. Supports minimization and maximization, square
and rectangular matrices, and integer (`i32`, `i64`, `isize`) or floating-point
(`f32`, `f64`) costs. Written in safe Rust.

## Installation

Requires Rust 1.85 or later. Use the GitHub repository as a Cargo dependency:

```toml
[dependencies]
jv = { package = "jv-rs", git = "https://github.com/ReinoutA/jv-rs", branch = "main" }
```

The Cargo package is named `jv-rs`; the library is imported as `jv`.
This project is not published on crates.io.

## Usage

```rust
use jv::DenseMatrix;

let costs = DenseMatrix::from_rows(&[[4, 1, 3], [2, 0, 5], [3, 2, 2]]).unwrap();
let a = jv::minimize(&costs).unwrap();
assert_eq!(a.cost, 5);

let b = jv::maximize(&costs).unwrap();
assert_eq!(b.cost, 11);
```

Every successful solve assigns exactly `min(rows, cols)` pairs. Each row and
column participates at most once; surplus rows or columns remain unassigned.
Empty matrices are supported. `row_to_col` and `col_to_row` contain `None` for
unassigned entries.

Borrow existing row-major data with `DenseMatrix::from_slice`, or transfer an
owned vector with `DenseMatrix::from_vec`. For repeated minimization, reuse
`JvSolver` and its work buffers:

```rust
use jv::{DenseMatrix, JvSolver, NoControl, Solver};

let costs = DenseMatrix::from_rows(&[[4, 1, 3], [2, 0, 5], [3, 2, 2]]).unwrap();
let mut solver = JvSolver::new();
let result = solver.solve(&costs, &mut NoControl).unwrap();
jv::verify::certify(&costs, &result, 0).unwrap();
```

`CostMatrix` supports custom matrix views. Noncontiguous and tall inputs are
copied into a reusable row-major work buffer. `Lap` provides objective selection
and a `Control` hook for cancellation. The hook runs before shortest-path
augmentations; initialization and row reduction are not interruptible, and a
solve completed during initialization may never call the hook.

## Numeric limits and verification

Costs must satisfy `|cost| <= Cost::max_abs(min(rows, cols))`. NaN, infinity, and
out-of-range costs return `LapError::InvalidCost` with their original matrix
coordinates. All finite edges must be present; forbidden or infinite-cost edges
are not supported. Integer costs give exact arithmetic within the accepted range;
floating-point calculations are subject to rounding.

Results include dual variables. For minimization, `u[row] + v[col] <= cost`,
with equality at assigned pairs; maximization reverses the inequality.
`verify::certify` checks minimization results with `rows <= cols`. Use zero
tolerance for integers and a finite nonnegative tolerance appropriate to the
scale of floating-point data. For tall problems, transpose both the matrix and
assignment before certification. The verifier rejects malformed mappings,
non-finite values, invalid tolerances, and arithmetic overflow.

## Development

```text
cargo +stable fmt --all -- --check
cargo +stable clippy --all-targets --all-features -- -D warnings
cargo +stable test --all-targets --all-features
cargo +stable test --doc
cargo +stable doc --no-deps
cargo +stable package
```

CI tests stable Rust on Linux, Windows, and macOS, and the minimum supported
Rust version on Linux. It also runs the ignored 200,000-instance stress test
with release overflow checks and debug assertions enabled. The regular tests
cover brute-force optima, dual certificates, matrix shapes, numerical limits,
solver reuse, and cancellation.

Generate local API documentation with `cargo doc --open`.

## License

A license has not been selected for this repository. crates.io publishing is
disabled until that decision is made.
