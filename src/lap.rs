use crate::algo::JvSolver;
use crate::matrix::Negated;
use crate::{Algorithm, Assignment, Control, Cost, CostMatrix, LapError, NoControl, Sense, Solver};

/// Find an assignment with minimal total cost, using default settings.
///
/// ```
/// use jv::DenseMatrix;
///
/// let costs = DenseMatrix::from_rows(&[[4, 1, 3], [2, 0, 5], [3, 2, 2]]).unwrap();
/// let a = jv::minimize(&costs).unwrap();
/// assert_eq!(a.cost, 5);
/// assert_eq!(a.row_to_col, [Some(1), Some(0), Some(2)]);
/// ```
pub fn minimize<M: CostMatrix>(costs: &M) -> Result<Assignment<M::Cost>, LapError> {
    Lap::new().minimize().solve(costs)
}

/// Find an assignment with maximal total value, using default settings.
///
/// The returned duals satisfy `u[i] + v[j] >= c[i][j]`.
///
/// ```
/// use jv::DenseMatrix;
///
/// let profits = DenseMatrix::from_rows(&[[1.0, 9.0], [8.0, 2.0], [5.0, 5.0]]).unwrap();
/// let a = jv::maximize(&profits).unwrap();
/// assert_eq!(a.cost, 17.0);
/// assert_eq!(a.row_to_col, [Some(1), Some(0), None]);
/// ```
pub fn maximize<M: CostMatrix>(costs: &M) -> Result<Assignment<M::Cost>, LapError> {
    Lap::new().maximize().solve(costs)
}

/// Configurable front end, for when the defaults of [`minimize`] and
/// [`maximize`] are not enough.
///
/// ```
/// use std::ops::ControlFlow;
/// use jv::{Algorithm, DenseMatrix, Lap};
///
/// let costs = DenseMatrix::from_rows(&[[4, 1, 3], [2, 0, 5], [3, 2, 2]]).unwrap();
/// let mut no_limit = || ControlFlow::Continue(());
/// let a = Lap::new()
///     .maximize()
///     .algorithm(Algorithm::Jv)
///     .solve_with(&costs, &mut no_limit)
///     .unwrap();
/// assert_eq!(a.cost, 11);
/// ```
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Lap {
    sense: Sense,
    algorithm: Algorithm,
}

impl Lap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn sense(mut self, sense: Sense) -> Self {
        self.sense = sense;
        self
    }

    pub fn minimize(self) -> Self {
        self.sense(Sense::Minimize)
    }

    pub fn maximize(self) -> Self {
        self.sense(Sense::Maximize)
    }

    pub fn algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    pub fn solve<M: CostMatrix>(&self, costs: &M) -> Result<Assignment<M::Cost>, LapError> {
        self.solve_with(costs, &mut NoControl)
    }

    /// Solve with a [`Control`] hook for cancellation or limits.
    ///
    /// For maximisation the returned duals satisfy `u[i] + v[j] >= c[i][j]`.
    pub fn solve_with<M, K>(
        &self,
        costs: &M,
        control: &mut K,
    ) -> Result<Assignment<M::Cost>, LapError>
    where
        M: CostMatrix,
        K: Control,
    {
        let mut solver = match self.algorithm {
            Algorithm::Auto | Algorithm::Jv => JvSolver::new(),
        };
        match self.sense {
            Sense::Minimize => solver.solve(costs, control),
            Sense::Maximize => {
                // Validate the original values before negating: integer MIN
                // cannot be negated, even though it is an invalid input cost.
                let pairs = costs.rows().min(costs.cols());
                for row in 0..costs.rows() {
                    for col in 0..costs.cols() {
                        if !costs.get(row, col).is_valid(pairs) {
                            return Err(LapError::InvalidCost { row, col });
                        }
                    }
                }
                let mut a = solver.solve(&Negated(costs), control)?;
                a.cost = a.cost.neg();
                if let Some(d) = &mut a.duals {
                    d.u.iter_mut()
                        .chain(d.v.iter_mut())
                        .for_each(|x| *x = x.neg());
                }
                Ok(a)
            }
        }
    }
}
