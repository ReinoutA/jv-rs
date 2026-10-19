use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LapError {
    /// Data length does not match the declared shape.
    ShapeMismatch { expected: usize, found: usize },
    /// A cost is NaN, infinite or larger than [`Cost::max_abs`](crate::Cost::max_abs).
    InvalidCost { row: usize, col: usize },
    /// No complete assignment exists (e.g. due to forbidden edges).
    Infeasible,
    /// The [`Control`](crate::Control) hook requested a stop.
    Cancelled,
}

impl fmt::Display for LapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, found } => {
                write!(
                    f,
                    "shape mismatch: expected {expected} elements, found {found}"
                )
            }
            Self::InvalidCost { row, col } => write!(f, "invalid cost at ({row}, {col})"),
            Self::Infeasible => write!(f, "no feasible assignment exists"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::error::Error for LapError {}
