//! Shared reconstruction infrastructure (architecture, damage/augmentation
//! utilities, metrics, sample-grid visualization) used by the Recon Train tab
//! (`recon_simple`) and the Pipeline's hole detection. The GAN trainer and its
//! "GAN Train" tab that used to live here were removed (legacy, unused) —
//! `model`/`training`/`metrics`/`visualization` remain because they're shared,
//! not GAN-specific: `model` owns the UNet architecture + device/backend
//! helpers every recon-related tab depends on, `training` owns the damage/
//! augmentation/loss utilities `recon_simple`'s trainer and the Pipeline's
//! hole detection both call directly, `metrics` is generic IoU/Dice/precision/
//! recall computation, and `visualization` is the shared sample-grid builder.

pub mod model;
pub mod metrics;
pub mod visualization;
pub mod training;
