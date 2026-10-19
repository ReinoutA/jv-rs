# jv-rs

A Rust library for linear assignment problems, based on
the Jonker–Volgenant algorithm. Supports minimization and maximization, square
and rectangular matrices, and integer (`i32`, `i64`, `isize`) or floating-point
(`f32`, `f64`) costs.

## Installation

Requires Rust 1.85 or later. Use the GitHub repository as a Cargo dependency:

```toml
[dependencies]
jv = { package = "jv-rs", git = "https://github.com/ReinoutA/jv-rs", branch = "main" }
```

The Cargo package is named `jv-rs`; the library is imported as `jv`.

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
