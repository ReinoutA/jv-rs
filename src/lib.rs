//! Linear assignment problem (LAP) solvers.
//!
//! Given an `n × m` cost matrix, find an assignment of rows to columns
//! (each row and each column used at most once, `min(n, m)` pairs in total)
//! that minimises (or maximises) the total cost.
//!
//! # Layers
//!
//! * [`Cost`]: scalar cost type (integers and floats).
//! * [`CostMatrix`]: anything that looks like a cost matrix; [`DenseMatrix`] is the
//!   built-in row-major implementation.
//! * [`Solver`]: a minimisation algorithm; [`JvSolver`] is the dense Jonker–Volgenant solver.
//! * [`Assignment`]: the result, including dual variables that certify optimality.
//! * [`minimize`], [`maximize`] and [`Lap`]: the front end (minimise / maximise, algorithm choice).

#![forbid(unsafe_code)]

pub mod algo;
pub mod control;
pub mod cost;
pub mod error;
mod lap;
pub mod matrix;
pub mod problem;
pub mod solution;
pub mod verify;

pub use algo::{Algorithm, JvSolver, Solver};
pub use control::{Control, NoControl};
pub use cost::Cost;
pub use error::LapError;
pub use lap::{Lap, maximize, minimize};
pub use matrix::{CostMatrix, DenseMatrix};
pub use problem::Sense;
pub use solution::{Assignment, Duals};
