use thiserror::Error;

#[derive(Debug, Error)]
pub enum SplineError {
    #[error("At least two vertices are required")]
    TooFewVertices,
    #[error("Invalid bc_type. Use 'natural', 'closed', or 'clamped'.")]
    InvalidBoundaryCondition,
    #[error("Grid must have same length as vertices")]
    GridLengthMismatch,
    #[error("Internal error in tangent calculation")]
    TangentError,
    #[error("Exactly 2 vertices are needed for a straight line")]
    StraightLineError,
    #[error("Invalid input dimensions")]
    InvalidDimensions,
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
    #[error("tolerance must be positive")]
    InvalidTolerance,
}
