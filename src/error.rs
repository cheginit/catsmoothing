use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SplineError {
    #[error("At least two vertices are required")]
    TooFewVertices,
    #[error("Invalid bc_type. Use 'natural', 'closed', or 'clamped'.")]
    InvalidBoundaryCondition,
    #[error("Grid must have same length as vertices")]
    GridLengthMismatch,
    #[error("Either n_pts or distance must be provided")]
    InvalidSmoothingParameters,
    #[error("Mismatched input lengths")]
    MismatchedInputLengths,
    #[error("Unsupported derivative order: only 0, 1, and 2 are supported")]
    UnsupportedDerivativeOrder,
    #[error("Vertices must be finite")]
    NonFiniteVertices,
    #[error("Grid must be finite and strictly increasing")]
    InvalidGrid,
    #[error("alpha must be finite")]
    InvalidAlpha,
    #[error("gaussian_sigma must be finite and non-negative")]
    InvalidGaussianSigma,
    #[error("n_pts must be at least 2")]
    InvalidPointCount,
    #[error("distance must be finite and positive")]
    InvalidDistance,
    #[error("Too many output points requested")]
    TooManyPoints,
    #[error("Invalid sigma_units. Use 'vertices' or 'length'.")]
    InvalidSigmaUnits,
    #[error("taubin_iterations must be at least 1")]
    InvalidTaubinIterations,
    #[error("gaussian_sigma and taubin_iterations cannot be combined")]
    ConflictingPresmoothing,
    #[error("Invalid continuity. Use 'c1', 'g2', or 'c2'.")]
    InvalidContinuity,
}
