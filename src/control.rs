use std::ops::ControlFlow;

/// Hook called periodically by solvers (once per augmentation).
///
/// Use it for cancellation, time limits or progress reporting.
/// [`NoControl`] compiles away entirely.
pub trait Control {
    fn tick(&mut self) -> ControlFlow<()>;
}

/// Never interrupts the solver.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoControl;

impl Control for NoControl {
    #[inline(always)]
    fn tick(&mut self) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

impl<F: FnMut() -> ControlFlow<()>> Control for F {
    #[inline]
    fn tick(&mut self) -> ControlFlow<()> {
        self()
    }
}
