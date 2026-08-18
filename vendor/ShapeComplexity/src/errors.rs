use thiserror::Error;
use std::io;
use std::path::PathBuf;

/// Custom error types for LeafComplexR
#[derive(Error, Debug)]
pub enum LeafComplexError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Image processing error: {0}")]
    Image(#[from] image::ImageError),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Failed to load configuration from {path}: {source}")]
    ConfigLoad {
        source: toml::de::Error,
        path: PathBuf,
    },

    #[error("Invalid reference point choice: {0}")]
    InvalidReferencePoint(String),

    #[error("Morphology error: {0}")]
    Morphology(String),

    #[error("Path algorithm error: {0}")]
    PathAlgorithm(String),

    #[error("CSV output error: {0}")]
    CsvOutput(#[from] csv::Error),

    #[error("No valid points found in image")]
    NoValidPoints,

    #[error("Invalid input path: {0}")]
    InvalidPath(PathBuf),
    
    #[error("Unexpected error: {0}")]
    Other(String),
}

/// Type alias for Result with our custom error type
pub type Result<T> = std::result::Result<T, LeafComplexError>;