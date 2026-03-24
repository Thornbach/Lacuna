use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        GroupNorm, GroupNormConfig, Linear, LinearConfig,
        PaddingConfig2d,
    },
    tensor::{Tensor, backend::Backend, activation},
};

// ── Re-export backend aliases from recon_train ────────────────────────────────
pub use crate::tabs::recon_train::model::{
    TrainBackend, InferBackend,
    create_train_device, create_infer_device, backend_name,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// LeakyReLU(0.2)
#[inline]
fn leaky_relu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let pos = x.clone().clamp_min(0.0f32);
    let neg = x.clamp_max(0.0f32).mul_scalar(0.2f32);
    pos + neg
}

// ── Leaf Area Regressor ───────────────────────────────────────────────────────
//
// Input:  [B, 5, H, W]   (RGBA normalised to [-1,1] + species channel = 1.0)
// Output: [B, 1]         (sigmoid scalar — predicted_total_leaf_fraction ∈ (0,1))
//
// Architecture:
//   c1: Conv2d  5→ 32, k=7 s=2 p=3, GroupNorm(4, 32),  LeakyReLU(0.2)   H→H/2
//   c2: Conv2d 32→ 64, k=4 s=2 p=1, GroupNorm(8, 64),  LeakyReLU(0.2)   H→H/4
//   c3: Conv2d 64→128, k=4 s=2 p=1, GroupNorm(8,128),  LeakyReLU(0.2)   H→H/8
//   c4: Conv2d128→256, k=4 s=2 p=1, GroupNorm(8,256),  LeakyReLU(0.2)   H→H/16
//   c5: Conv2d256→256, k=4 s=2 p=1, GroupNorm(8,256),  LeakyReLU(0.2)   H→H/32
//   Global avg pool: mean_dim(3).mean_dim(2) → [B, 256, 1, 1]
//   Reshape → [B, 256]
//   fc1: Linear 256→64, ReLU
//   fc2: Linear  64→ 1
//   sigmoid → [B, 1]

#[derive(Module, Debug)]
pub struct LeafAreaRegressor<B: Backend> {
    c1: Conv2d<B>, n1: GroupNorm<B>,
    c2: Conv2d<B>, n2: GroupNorm<B>,
    c3: Conv2d<B>, n3: GroupNorm<B>,
    c4: Conv2d<B>, n4: GroupNorm<B>,
    c5: Conv2d<B>, n5: GroupNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> LeafAreaRegressor<B> {
    pub fn init(device: &B::Device) -> Self {
        // Conv helper: stride-2 downsampling
        let conv7 = |ic, oc| Conv2dConfig::new([ic, oc], [7, 7])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(3, 3))
            .with_bias(false);
        let conv4 = |ic, oc| Conv2dConfig::new([ic, oc], [4, 4])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false);

        Self {
            c1: conv7(5,   32).init(device), n1: GroupNormConfig::new(4,  32).init(device),
            c2: conv4(32,  64).init(device), n2: GroupNormConfig::new(8,  64).init(device),
            c3: conv4(64, 128).init(device), n3: GroupNormConfig::new(8, 128).init(device),
            c4: conv4(128,256).init(device), n4: GroupNormConfig::new(8, 256).init(device),
            c5: conv4(256,256).init(device), n5: GroupNormConfig::new(8, 256).init(device),
            fc1: LinearConfig::new(256, 64).init(device),
            fc2: LinearConfig::new(64,   1).init(device),
        }
    }

    /// GPU-CPU fence — reads the first conv weight to drain pending commands.
    pub fn gpu_fence(&self) -> Tensor<B, 1> {
        self.c1.weight.val().flatten::<1>(0, 3).mean()
    }

    /// Forward pass. Returns [B, 1] sigmoid predictions in (0, 1).
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        // ── Convolutional backbone ─────────────────────────────────────────────
        let x = leaky_relu(self.n1.forward(self.c1.forward(x)));
        let x = leaky_relu(self.n2.forward(self.c2.forward(x)));
        let x = leaky_relu(self.n3.forward(self.c3.forward(x)));
        let x = leaky_relu(self.n4.forward(self.c4.forward(x)));
        let x = leaky_relu(self.n5.forward(self.c5.forward(x)));

        // ── Global average pooling: [B, 256, H, W] → [B, 256, 1, 1] ──────────
        let x = x.mean_dim(3).mean_dim(2);   // W-dim first, then H-dim

        // ── Flatten: [B, 256, 1, 1] → [B, 256] ───────────────────────────────
        let [b, c, _, _] = x.dims();
        let x = x.reshape([b, c]);

        // ── Fully-connected head ───────────────────────────────────────────────
        let x = activation::relu(self.fc1.forward(x));
        let x = self.fc2.forward(x);

        // ── Sigmoid output ─────────────────────────────────────────────────────
        activation::sigmoid(x)
    }
}
