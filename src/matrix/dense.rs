use std::borrow::Cow;

use super::CostMatrix;
use crate::{Cost, LapError};

/// Dense row-major cost matrix, borrowed or owned.
#[derive(Debug, Clone)]
pub struct DenseMatrix<'a, C: Cost> {
    data: Cow<'a, [C]>,
    rows: usize,
    cols: usize,
}

impl<'a, C: Cost> DenseMatrix<'a, C> {
    /// Borrow a row-major slice of length `rows * cols`.
    pub fn from_slice(data: &'a [C], rows: usize, cols: usize) -> Result<Self, LapError> {
        check_len(data.len(), rows, cols)?;
        Ok(Self {
            data: Cow::Borrowed(data),
            rows,
            cols,
        })
    }

    /// Take ownership of a row-major vector of length `rows * cols`.
    pub fn from_vec(data: Vec<C>, rows: usize, cols: usize) -> Result<Self, LapError> {
        check_len(data.len(), rows, cols)?;
        Ok(Self {
            data: Cow::Owned(data),
            rows,
            cols,
        })
    }

    pub fn as_slice(&self) -> &[C] {
        &self.data
    }
}

impl<C: Cost> DenseMatrix<'static, C> {
    /// Build from nested rows; all rows must have equal length.
    pub fn from_rows<R: AsRef<[C]>>(rows: &[R]) -> Result<Self, LapError> {
        let cols = rows.first().map_or(0, |r| r.as_ref().len());
        let mut data = Vec::with_capacity(rows.len() * cols);
        for r in rows {
            let r = r.as_ref();
            if r.len() != cols {
                return Err(LapError::ShapeMismatch {
                    expected: cols,
                    found: r.len(),
                });
            }
            data.extend_from_slice(r);
        }
        Self::from_vec(data, rows.len(), cols)
    }
}

fn check_len(len: usize, rows: usize, cols: usize) -> Result<(), LapError> {
    match rows.checked_mul(cols) {
        Some(expected) if expected == len => Ok(()),
        Some(expected) => Err(LapError::ShapeMismatch {
            expected,
            found: len,
        }),
        None => Err(LapError::ShapeMismatch {
            expected: usize::MAX,
            found: len,
        }),
    }
}

impl<C: Cost> CostMatrix for DenseMatrix<'_, C> {
    type Cost = C;

    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    #[inline]
    fn get(&self, row: usize, col: usize) -> C {
        self.data[row * self.cols + col]
    }
    fn as_row_major(&self) -> Option<&[C]> {
        Some(&self.data)
    }
}
