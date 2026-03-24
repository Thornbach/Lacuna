/// Radial profile completion network.
///
/// # Task
/// Given a 1-D radial distance profile sampled at N equally-spaced angles
/// around the leaf centroid, predict the completed (intact) profile.
///
/// # Why this works for oak leaves
/// Oak leaf lobes are star-shaped from the centroid: every boundary point is
/// visible from the centre without occlusion.  Lobe damage therefore appears
/// as a local *dip* in the radial profile r(θ).  The network must fill in
/// that dip using the intact lobes on either side — a purely 1-D signal
/// completion task with ~200 k parameters and deterministic gradients.
///
/// Interior damage (holes, spots) is *invisible* to the radial profile — the
/// outermost leaf pixel at each angle is unchanged.  This is the correct
/// behaviour: interior damage does not affect leaf area.
///
/// # Architecture
/// 1-D U-Net with 4 encoder levels and bottleneck self-attention.
///
/// ```text
/// Input  [B, 3, N]   r_normalized ∈ [0,1]  |  valid_flag ∈ {0,1}  |  r_interp ∈ [0,1]
///   Ch0  damaged profile (0 where missing)
///   Ch1  valid flag (1 where ray hit the leaf, 0 in damaged gap)
///   Ch2  gap-filled interpolation: circular linear interp from valid neighbours
///
/// e1 Conv1d(2 → 32,  k=7, s=2, pad=3)  → [B, 32, N/2]
/// e2 Conv1d(32 → 64, k=7, s=2, pad=3)  → [B, 64, N/4]   + GN(8)
/// e3 Conv1d(64 →128, k=7, s=2, pad=3)  → [B,128, N/8]   + GN(8)
/// e4 Conv1d(128→256, k=7, s=2, pad=3)  → [B,256, N/16]  + GN(8) ← bottleneck
///    → 1-D self-attention over N/16 positions
///
/// d4 ConvT1d(256→128, k=4,s=2,p=1) + GN + drop  [B,128, N/8]   (no skip)
/// d3 ConvT1d(256→ 64, k=4,s=2,p=1) + GN + drop  [B, 64, N/4]   (cat d4+e3=256)
/// d2 ConvT1d(128→ 32, k=4,s=2,p=1) + GN         [B, 32, N/2]   (cat d3+e2=128)
/// d1 ConvT1d( 64→  1, k=4,s=2,p=1, bias=true)   [B,  1, N  ]   (cat d2+e1= 64)
/// out Conv1d(1→1, k=3,s=1,p=1, bias=true) → ReLU                [B,  1, N  ]
/// ```
///
/// With N=512 default: bottleneck is 32 positions — self-attention attends
/// all 32 positions, spanning the full leaf boundary in one shot.
use burn::{
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig},
        Dropout, DropoutConfig,
        GroupNorm, GroupNormConfig, Linear, LinearConfig,
        PaddingConfig1d,
    },
    tensor::{Tensor, backend::Backend, activation},
};

use crate::tabs::recon_train::model::{TrainBackend, InferBackend};
pub use crate::tabs::recon_train::model::{create_train_device, create_infer_device, backend_name};

/// Number of radial rays (angular samples).  Must be a power of 2 ≥ 16.
pub const N_RAYS: usize = 512;

#[inline]
fn lrelu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone().clamp_min(0.0f32) + x.clamp_max(0.0f32).mul_scalar(0.2f32)
}

// ── RadialNet ─────────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct RadialNet<B: Backend> {
    // Encoder
    e1_conv: Conv1d<B>,
    e2_conv: Conv1d<B>, e2_gn: GroupNorm<B>,
    e3_conv: Conv1d<B>, e3_gn: GroupNorm<B>,
    e4_conv: Conv1d<B>, e4_gn: GroupNorm<B>,

    // Decoder
    d4_conv: ConvTranspose1d<B>, d4_gn: GroupNorm<B>,
    d3_conv: ConvTranspose1d<B>, d3_gn: GroupNorm<B>,
    d2_conv: ConvTranspose1d<B>, d2_gn: GroupNorm<B>,
    d1_conv: ConvTranspose1d<B>,
    out_conv: Conv1d<B>,

    drop: Dropout,

    // Bottleneck self-attention over sequence positions
    sa_q:    Linear<B>,  // 256 → 64
    sa_k:    Linear<B>,  // 256 → 64
    sa_v:    Linear<B>,  // 256 → 256
    sa_proj: Linear<B>,  // 256 → 256
}

impl<B: Backend> RadialNet<B> {
    pub fn init(device: &B::Device) -> Self {
        // Conv1dConfig::new(channels_in, channels_out, kernel_size) — 3 scalar args in burn 0.16
        // PaddingConfig1d::Explicit(pad) — single symmetric value
        let e = |ic, oc| Conv1dConfig::new(ic, oc, 7)
            .with_stride(2)
            .with_padding(PaddingConfig1d::Explicit(3))
            .with_bias(false);
        // ConvTranspose1dConfig::new([channels_in, channels_out], kernel_size)
        let d = |ic, oc| ConvTranspose1dConfig::new([ic, oc], 4)
            .with_stride(2)
            .with_padding(1)
            .with_bias(false);
        let gn = |ch| GroupNormConfig::new(8, ch);

        Self {
            e1_conv: e(3,    32).init(device),
            e2_conv: e(32,   64).init(device), e2_gn: gn(64).init(device),
            e3_conv: e(64,  128).init(device), e3_gn: gn(128).init(device),
            e4_conv: e(128, 256).init(device), e4_gn: gn(256).init(device),

            d4_conv: d(256, 128).init(device), d4_gn: gn(128).init(device),
            d3_conv: d(256,  64).init(device), d3_gn: gn(64).init(device),
            d2_conv: d(128,  32).init(device), d2_gn: gn(32).init(device),
            d1_conv: ConvTranspose1dConfig::new([64, 1], 4)
                        .with_stride(2)
                        .with_padding(1)
                        .with_bias(true)
                        .init(device),
            out_conv: Conv1dConfig::new(1, 1, 3)
                        .with_padding(PaddingConfig1d::Explicit(1))
                        .with_bias(true)
                        .init(device),

            drop: DropoutConfig::new(0.5).init(),

            sa_q:    LinearConfig::new(256, 64).init(device),
            sa_k:    LinearConfig::new(256, 64).init(device),
            sa_v:    LinearConfig::new(256, 256).init(device),
            sa_proj: LinearConfig::new(256, 256).init(device),
        }
    }

    pub fn gpu_fence(&self) -> Tensor<B, 1> {
        self.e1_conv.weight.val().flatten::<1>(0, 2).mean()
    }

    fn bottleneck_attn(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, c, n] = x.dims();
        let seq = x.clone().swap_dims(1, 2);   // [B, N, C]
        let q = self.sa_q.forward(seq.clone());
        let k = self.sa_k.forward(seq.clone());
        let v = self.sa_v.forward(seq.clone());
        let scale = (64.0_f32).sqrt();
        let attn  = activation::softmax(
            q.matmul(k.swap_dims(1, 2)).div_scalar(scale), 2
        );
        let proj = self.sa_proj.forward(attn.matmul(v));
        x + proj.swap_dims(1, 2).reshape([b, c, n])
    }

    fn encode(&self, x: Tensor<B, 3>)
        -> (Tensor<B,3>, Tensor<B,3>, Tensor<B,3>, Tensor<B,3>)
    {
        let e1 = lrelu(self.e1_conv.forward(x));
        let e2 = lrelu(self.e2_gn.forward(self.e2_conv.forward(e1.clone())));
        let e3 = lrelu(self.e3_gn.forward(self.e3_conv.forward(e2.clone())));
        let e4 = lrelu(self.e4_gn.forward(self.e4_conv.forward(e3.clone())));
        let e4 = self.bottleneck_attn(e4);
        (e1, e2, e3, e4)
    }

    fn decode(
        &self,
        e1: Tensor<B,3>, e2: Tensor<B,3>, e3: Tensor<B,3>, e4: Tensor<B,3>,
    ) -> Tensor<B, 3> {
        // d4: expand bottleneck — no skip concat
        let d4 = self.drop.forward(
            activation::relu(self.d4_gn.forward(self.d4_conv.forward(e4)))
        );
        // d3: cat(d4=128, e3=128) = 256 → 64
        let d3 = self.drop.forward(
            activation::relu(self.d3_gn.forward(
                self.d3_conv.forward(Tensor::cat(vec![d4, e3], 1))
            ))
        );
        // d2: cat(d3=64, e2=64) = 128 → 32
        let d2 = activation::relu(self.d2_gn.forward(
            self.d2_conv.forward(Tensor::cat(vec![d3, e2], 1))
        ));
        // d1: cat(d2=32, e1=32) = 64 → 1
        let d1 = self.d1_conv.forward(Tensor::cat(vec![d2, e1], 1));
        // out: smooth + ReLU (radius must be ≥ 0)
        activation::relu(self.out_conv.forward(d1))
    }

    /// Forward: input [B, 2, N] → predicted radii [B, 1, N] ≥ 0.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let (e1, e2, e3, e4) = self.encode(x);
        self.decode(e1, e2, e3, e4)
    }
}

// ── Checkpoint helpers ────────────────────────────────────────────────────────

use burn_core::record::CompactRecorder;
use burn_core::module::Module as CoreModule;
use burn::module::AutodiffModule;
use std::path::Path;

pub fn save_radial_checkpoint(
    model: &RadialNet<TrainBackend>,
    dir:   &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let _: f32 = model.gpu_fence().into_scalar();
    let rec = CompactRecorder::new();
    model.clone().valid()
        .save_file(dir.join("radial"), &rec)
        .map_err(|e| format!("{e}"))
}

pub fn load_radial_checkpoint(
    model:  RadialNet<TrainBackend>,
    dir:    &Path,
    device: &<TrainBackend as burn::tensor::backend::Backend>::Device,
) -> Result<RadialNet<TrainBackend>, String> {
    let rec = CompactRecorder::new();
    model.load_file(dir.join("radial"), &rec, device)
        .map_err(|e| format!("{e}"))
}

pub fn load_radial_infer(
    dir:    &Path,
    device: &<InferBackend as burn::tensor::backend::Backend>::Device,
) -> Result<RadialNet<InferBackend>, String> {
    let rec = CompactRecorder::new();
    RadialNet::<InferBackend>::init(device)
        .load_file(dir.join("radial"), &rec, device)
        .map_err(|e| format!("{e}"))
}

// ── Radial geometry ───────────────────────────────────────────────────────────

/// Extract the radial boundary profile from a binary alpha mask.
///
/// For each of `N_RAYS` evenly-spaced angles, marches a ray from the centroid
/// outward and records the normalised distance to the outermost opaque pixel.
///
/// Returns `(profile, valid, cx, cy)`:
/// - `profile[i]` — normalised radius ∈ [0, 1] (0 = centroid, 1 = max possible r)
/// - `valid[i]`   — true if the ray intersects the leaf at all
/// - `cx`, `cy`   — centroid pixel coordinates
pub fn extract_radial(
    alpha: &[bool],
    w:     usize,
    h:     usize,
) -> (Vec<f32>, Vec<bool>, f32, f32) {
    let n = N_RAYS;

    // Centroid of leaf pixels
    let (mut sx, mut sy, mut cnt) = (0f64, 0f64, 0usize);
    for y in 0..h {
        for x in 0..w {
            if alpha[y * w + x] { sx += x as f64; sy += y as f64; cnt += 1; }
        }
    }
    let (cx, cy) = if cnt > 0 {
        ((sx / cnt as f64) as f32, (sy / cnt as f64) as f32)
    } else {
        (w as f32 / 2.0, h as f32 / 2.0)
    };

    let max_r = ((w.min(h)) as f32 / 2.0).max(1.0);
    let steps = (max_r * 2.0 + 2.0) as usize;

    let mut profile = vec![0.0f32; n];
    let mut valid   = vec![false;  n];

    for i in 0..n {
        let angle = std::f32::consts::TAU * (i as f32) / (n as f32);
        let (dx, dy) = (angle.cos(), angle.sin());
        for s in 0..steps {
            let t  = s as f32 * 0.5;
            let px = (cx + dx * t).round() as isize;
            let py = (cy + dy * t).round() as isize;
            if px < 0 || px >= w as isize || py < 0 || py >= h as isize { break; }
            if alpha[py as usize * w + px as usize] {
                let r = t / max_r;
                if r > profile[i] { profile[i] = r; }
                valid[i] = true;
            }
        }
    }

    (profile, valid, cx, cy)
}

/// Reconstruct a binary alpha mask from a predicted radial profile.
///
/// For each pixel (x,y), converts to polar coordinates relative to the centroid
/// and marks it as leaf if its distance is within the (linearly interpolated)
/// predicted radius at that angle.
pub fn fill_from_radial(
    profile: &[f32],
    w:       usize,
    h:       usize,
    cx:      f32,
    cy:      f32,
) -> Vec<bool> {
    let n     = profile.len();
    let max_r = ((w.min(h)) as f32 / 2.0).max(1.0);
    let mut mask = vec![false; w * h];
    let tau = std::f32::consts::TAU;

    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let r  = (dx * dx + dy * dy).sqrt();
            // Angle in [0, 2π)
            let mut angle = dy.atan2(dx);
            if angle < 0.0 { angle += tau; }
            // Interpolate between adjacent profile samples
            let t   = (angle / tau) * n as f32;
            let i0  = t as usize % n;
            let i1  = (i0 + 1) % n;
            let frc = t - t.floor();
            let r_thresh = (profile[i0] * (1.0 - frc) + profile[i1] * frc) * max_r;
            if r <= r_thresh { mask[y * w + x] = true; }
        }
    }
    mask
}
