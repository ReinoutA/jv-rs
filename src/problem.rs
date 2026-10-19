/// Optimisation direction.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Sense {
    #[default]
    Minimize,
    Maximize,
}
