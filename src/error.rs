use thiserror::Error;

#[derive(Debug, Error)]
pub enum SplineError {
    #[error("At least two vertices are required")]
    TooFewVertices,
    #[error("Invalid boundary condition type")]
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
}
