use std::fmt::Debug;
use std::ops::{Add, Sub};

/// Scalar cost type.
///
/// Integers are supported next to floats because they give exact results.
/// To rule out overflow, solvers only accept costs with
/// `|c| <= Cost::max_abs(pairs)`, where `pairs = min(rows, cols)`; see
/// [`Cost::max_abs`]. NaN is always rejected.
pub trait Cost: Copy + PartialOrd + Add<Output = Self> + Sub<Output = Self> + Debug {
    const ZERO: Self;
    const MAX: Self;

    /// Largest accepted `|cost|` for a problem with `pairs` assigned pairs.
    /// Integers use `MAX / max(16, pairs)`. Floats use
    /// `MAX / max(16, 2 * pairs)` to reserve accumulation-rounding headroom,
    /// with `MAX` the largest finite value in both formulas.
    ///
    /// Rationale (with `C` the largest `|cost|`): every assigned row uses its
    /// cheapest reduced cost, so an assigned column's price is at most `2C`
    /// below any other price, and an unassigned column with price in `[-C, C]`
    /// exists until the last assignment. Prices therefore stay within
    /// `[-5C, C]` (the `-5C` only after the final augmenting row reduction
    /// step), intermediate values within `14C`, and the total cost is at most
    /// `pairs * C`.
    fn max_abs(pairs: usize) -> Self;

    /// `true` if the value is usable as a cost in a problem with `pairs` pairs.
    fn is_valid(self, pairs: usize) -> bool {
        let max = Self::max_abs(pairs);
        self <= max && self >= max.neg()
    }

    /// Whether this value is finite. NaN and infinities are rejected.
    fn is_finite(self) -> bool;

    /// Add without overflow or a non-finite result.
    /// Bounded integer implementations must override this default.
    fn checked_add(self, rhs: Self) -> Option<Self> {
        let sum = self + rhs;
        sum.is_finite().then_some(sum)
    }

    /// Subtract without overflow or a non-finite result.
    /// Bounded integer implementations must override this default.
    fn checked_sub(self, rhs: Self) -> Option<Self> {
        let difference = self - rhs;
        difference.is_finite().then_some(difference)
    }

    /// Arithmetic negation, used to turn maximisation into minimisation.
    fn neg(self) -> Self {
        Self::ZERO - self
    }
}

macro_rules! impl_cost_int {
    ($($t:ty),*) => {$(
        impl Cost for $t {
            const ZERO: Self = 0;
            const MAX: Self = <$t>::MAX;

            fn max_abs(pairs: usize) -> Self {
                (<$t>::MAX as u128 / pairs.max(16) as u128) as $t
            }

            fn is_finite(self) -> bool {
                true
            }

            fn checked_add(self, rhs: Self) -> Option<Self> {
                <$t>::checked_add(self, rhs)
            }

            fn checked_sub(self, rhs: Self) -> Option<Self> {
                <$t>::checked_sub(self, rhs)
            }
        }
    )*};
}

macro_rules! impl_cost_float {
    ($($t:ty),*) => {$(
        impl Cost for $t {
            const ZERO: Self = 0.0;
            const MAX: Self = <$t>::INFINITY;

            fn max_abs(pairs: usize) -> Self {
                <$t>::MAX / (2.0 * pairs as $t).max(16.0)
            }

            fn is_finite(self) -> bool {
                <$t>::is_finite(self)
            }
        }
    )*};
}

impl_cost_int!(i32, i64, isize);
impl_cost_float!(f32, f64);
