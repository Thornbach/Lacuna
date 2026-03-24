use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        Dropout, DropoutConfig,
        Embedding, EmbeddingConfig,
        GroupNorm, GroupNormConfig, Linear, LinearConfig, PaddingConfig2d,
    },
    tensor::{Tensor, Int, backend::Backend, activation},
};

// ── Backend type aliases ───────────────────────────────────────────────────────
//
// Selected at compile time via Cargo features:
//   cargo build                              → CUDA  (default, Windows training)
//   cargo build --no-default-features \
//               --features wgpu-gpu          → wgpu  (Linux GPU demo)
//   cargo build --no-default-features        → NdArray CPU  (portable demo)

#[cfg(feature = "cuda")]
pub type TrainBackend = burn::backend::Autodiff<burn_cuda::Cuda>;
#[cfg(feature = "cuda")]
pub type InferBackend = burn_cuda::Cuda;

#[cfg(all(feature = "wgpu-gpu", not(feature = "cuda")))]
pub type TrainBackend = burn::backend::Autodiff<burn::backend::Wgpu>;
#[cfg(all(feature = "wgpu-gpu", not(feature = "cuda")))]
pub type InferBackend = burn::backend::Wgpu;

#[cfg(not(any(feature = "cuda", feature = "wgpu-gpu")))]
pub type TrainBackend = burn::backend::Autodiff<burn::backend::NdArray>;
#[cfg(not(any(feature = "cuda", feature = "wgpu-gpu")))]
pub type InferBackend = burn::backend::NdArray;

// ── Device creation ───────────────────────────────────────────────────────────

/// Create the training device for whichever backend was compiled in.
pub fn create_train_device() -> <TrainBackend as Backend>::Device {
    #[cfg(feature = "cuda")]
    { burn_cuda::CudaDevice::new(0) }

    #[cfg(all(feature = "wgpu-gpu", not(feature = "cuda")))]
    {
        #[allow(deprecated)]
        burn::backend::wgpu::WgpuDevice::BestAvailable
    }

    #[cfg(not(any(feature = "cuda", feature = "wgpu-gpu")))]
    { burn::backend::ndarray::NdArrayDevice::Cpu }
}

/// Create the inference device for whichever backend was compiled in.
pub fn create_infer_device() -> <InferBackend as Backend>::Device {
    #[cfg(feature = "cuda")]
    { burn_cuda::CudaDevice::new(0) }

    #[cfg(all(feature = "wgpu-gpu", not(feature = "cuda")))]
    {
        #[allow(deprecated)]
        burn::backend::wgpu::WgpuDevice::BestAvailable
    }

    #[cfg(not(any(feature = "cuda", feature = "wgpu-gpu")))]
    { burn::backend::ndarray::NdArrayDevice::Cpu }
}

/// Short name of the active backend — shown as a badge in the UI.
pub fn backend_name() -> &'static str {
    #[cfg(feature = "cuda")]             { "CUDA" }
    #[cfg(all(feature = "wgpu-gpu", not(feature = "cuda")))] { "wgpu" }
    #[cfg(not(any(feature = "cuda", feature = "wgpu-gpu")))] { "CPU"  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// LeakyReLU(0.2)
#[inline]
pub fn leaky_relu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let pos = x.clone().clamp_min(0.0f32);
    let neg = x.clamp_max(0.0f32).mul_scalar(0.2f32);
    pos + neg
}

// ── U-Net Generator ───────────────────────────────────────────────────────────
//
// Input:  [B, 4, H, W]   (R, G, B, A — no symmetry channel; Quercus is asymmetric)
// Output: [B, 1, H, W]   (raw logits; sigmoid applied by forward_probs())
//
// Encoder: 7 × stride-2 Conv2d(4×4)  →  4×4 bottleneck at H=512
// Decoder: 7 × upsample_nearest2x + Conv2d(3×3)  — zero artifact risk
//          Skip connections at every level.
// Norms:   GroupNorm(8) throughout — stable at any batch size
// Dropout: p=0.5 on d7/d6/d5 (disabled in .valid() mode)
// Attention: self-attention at bottleneck e7 (16 tokens → full-leaf context)
// FiLM:    species/margin embeddings → scale+bias on bottleneck
// Area head: GAP(e7) → FC → predicted leaf area fraction (auxiliary loss)

/// Number of leaf-shape classes (LeafShape enum variants).
pub const SHAPE_CLASSES:  usize = 6;
/// Number of margin-type classes (MarginType enum variants).
pub const MARGIN_CLASSES: usize = 4;
const EMB_DIM:  usize = 32;
const COND_DIM: usize = EMB_DIM * 2; // 64

/// Nearest-neighbour 2× upsample — artifact-free alternative to ConvTranspose2d.
pub fn upsample_nearest2x<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, c, h, w] = x.dims();
    x.reshape([b, c, h, 1, w, 1])
     .repeat_dim(3, 2)
     .repeat_dim(5, 2)
     .reshape([b, c, h * 2, w * 2])
}

#[derive(Module, Debug)]
pub struct UNetGenerator<B: Backend> {
    // Encoder (stride-2 Conv2d 4×4, GroupNorm on e2..e7)
    e1_conv: Conv2d<B>,
    e2_conv: Conv2d<B>, e2_gn: GroupNorm<B>,   //  64→128
    e3_conv: Conv2d<B>, e3_gn: GroupNorm<B>,   // 128→256
    e4_conv: Conv2d<B>, e4_gn: GroupNorm<B>,   // 256→512
    e5_conv: Conv2d<B>, e5_gn: GroupNorm<B>,   // 512→512
    e6_conv: Conv2d<B>, e6_gn: GroupNorm<B>,   // 512→512
    e7_conv: Conv2d<B>, e7_gn: GroupNorm<B>,   // 512→512  4×4 bottleneck @ 512px

    // Decoder (upsample_nearest2x + Conv2d 3×3, GroupNorm on d7..d2, dropout d7..d5)
    d7_conv: Conv2d<B>, d7_gn: GroupNorm<B>,   // up(512)→512            [no skip]
    d6_conv: Conv2d<B>, d6_gn: GroupNorm<B>,   // up(cat(512+512))→512
    d5_conv: Conv2d<B>, d5_gn: GroupNorm<B>,   // up(cat(512+512))→512
    d4_conv: Conv2d<B>, d4_gn: GroupNorm<B>,   // up(cat(512+512))→256
    d3_conv: Conv2d<B>, d3_gn: GroupNorm<B>,   // up(cat(256+256))→128
    d2_conv: Conv2d<B>, d2_gn: GroupNorm<B>,   // up(cat(128+128))→64
    d1_conv: Conv2d<B>,                         // up(cat(64+64))→1  bias=true
    out_conv: Conv2d<B>,                        //     1→1  k=3

    drop: Dropout,

    // Bottleneck self-attention (16 tokens @ 512px)
    sa_q:    Linear<B>,  // 512→64
    sa_k:    Linear<B>,  // 512→64
    sa_v:    Linear<B>,  // 512→512
    sa_proj: Linear<B>,  // 512→512

    // FiLM descriptor conditioning (applied to e7 after attention)
    shape_embed:  Embedding<B>,  // SHAPE_CLASSES × EMB_DIM
    margin_embed: Embedding<B>,  // MARGIN_CLASSES × EMB_DIM
    film_scale:   Linear<B>,     // COND_DIM → 512
    film_bias:    Linear<B>,     // COND_DIM → 512

    // Area head: GAP of e7 → predicted leaf-area fraction ∈ (0,1)
    area_fc1: Linear<B>,  // 512→64
    area_fc2: Linear<B>,  //  64→1
}

impl<B: Backend> UNetGenerator<B> {
    pub fn init(device: &B::Device) -> Self {
        // Encoder: stride-2 Conv2d(4×4, pad=1)
        let enc = |ic, oc| Conv2dConfig::new([ic, oc], [4, 4])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false);
        // Decoder: standard Conv2d(3×3, pad=1) after upsample — no stride artifacts
        let dec = |ic, oc| Conv2dConfig::new([ic, oc], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false);
        let gn = |ch| GroupNormConfig::new(8, ch);

        // Half-width schedule (256-channel max instead of 512). A binary silhouette
        // does not need a 512-wide bottleneck; halving cuts params ~4× in the wide
        // layers and activations ~2× → roughly half the VRAM for both train + infer.
        Self {
            e1_conv: enc(4,   32).init(device),
            e2_conv: enc(32,  64).init(device),  e2_gn: gn(64).init(device),
            e3_conv: enc(64,  128).init(device), e3_gn: gn(128).init(device),
            e4_conv: enc(128, 256).init(device), e4_gn: gn(256).init(device),
            e5_conv: enc(256, 256).init(device), e5_gn: gn(256).init(device),
            e6_conv: enc(256, 256).init(device), e6_gn: gn(256).init(device),
            e7_conv: enc(256, 256).init(device), e7_gn: gn(256).init(device),

            // d7: up(256)→256, no skip
            d7_conv: dec(256,  256).init(device), d7_gn: gn(256).init(device),
            // d6: up(cat(256+256)=512)→256
            d6_conv: dec(512,  256).init(device), d6_gn: gn(256).init(device),
            // d5: up(cat(256+256)=512)→256
            d5_conv: dec(512,  256).init(device), d5_gn: gn(256).init(device),
            // d4: up(cat(256+256)=512)→128
            d4_conv: dec(512,  128).init(device), d4_gn: gn(128).init(device),
            // d3: up(cat(128+128)=256)→64
            d3_conv: dec(256,  64).init(device),  d3_gn: gn(64).init(device),
            // d2: up(cat(64+64)=128)→32
            d2_conv: dec(128,  32).init(device),  d2_gn: gn(32).init(device),
            // d1: up(cat(32+32)=64)→1
            d1_conv: Conv2dConfig::new([64, 1], [3, 3])
                        .with_padding(PaddingConfig2d::Explicit(1, 1))
                        .with_bias(true)
                        .init(device),
            out_conv: Conv2dConfig::new([1, 1], [3, 3])
                        .with_padding(PaddingConfig2d::Explicit(1, 1))
                        .with_bias(true)
                        .init(device),

            drop: DropoutConfig::new(0.5).init(),

            sa_q:    LinearConfig::new(256, 64).init(device),
            sa_k:    LinearConfig::new(256, 64).init(device),
            sa_v:    LinearConfig::new(256, 256).init(device),
            sa_proj: LinearConfig::new(256, 256).init(device),

            shape_embed:  EmbeddingConfig::new(SHAPE_CLASSES, EMB_DIM).init(device),
            margin_embed: EmbeddingConfig::new(MARGIN_CLASSES, EMB_DIM).init(device),
            film_scale:   LinearConfig::new(COND_DIM, 256).init(device),
            film_bias:    LinearConfig::new(COND_DIM, 256).init(device),

            area_fc1: LinearConfig::new(256, 64).with_bias(true).init(device),
            area_fc2: LinearConfig::new(64,   1).with_bias(true).init(device),
        }
    }

    pub fn gpu_fence(&self) -> Tensor<B, 1> {
        self.e1_conv.weight.val().flatten::<1>(0, 3).mean()
    }

    fn bottleneck_attn(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();
        let n = h * w;
        let seq = x.clone().reshape([b, c, n]).swap_dims(1, 2); // [B, N, C]
        let q   = self.sa_q.forward(seq.clone());
        let k   = self.sa_k.forward(seq.clone());
        let v   = self.sa_v.forward(seq.clone());
        let scale = (64.0_f32).sqrt();
        let attn  = activation::softmax(q.matmul(k.swap_dims(1, 2)).div_scalar(scale), 2);
        let proj  = self.sa_proj.forward(attn.matmul(v));
        x + proj.swap_dims(1, 2).reshape([b, c, h, w])
    }

    fn film_condition(
        &self,
        x:          Tensor<B, 4>,
        shape_cls:  u32,
        margin_cls: u32,
    ) -> Tensor<B, 4> {
        let [b, c, _h, _w] = x.dims();
        let device = x.device();
        let shape_idx: Tensor<B, 2, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(vec![shape_cls as i32; b], [b, 1]), &device,
        );
        let margin_idx: Tensor<B, 2, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(vec![margin_cls as i32; b], [b, 1]), &device,
        );
        let shape_emb  = self.shape_embed.forward(shape_idx).reshape([b, EMB_DIM]);
        let margin_emb = self.margin_embed.forward(margin_idx).reshape([b, EMB_DIM]);
        let cond  = Tensor::cat(vec![shape_emb, margin_emb], 1); // [B, COND_DIM]
        let scale = self.film_scale.forward(cond.clone()).reshape([b, c, 1, 1]);
        let bias  = self.film_bias.forward(cond).reshape([b, c, 1, 1]);
        x * (scale.add_scalar(1.0)) + bias
    }

    fn encode(
        &self,
        x:          Tensor<B, 4>,
        shape_cls:  u32,
        margin_cls: u32,
    ) -> (Tensor<B,4>, Tensor<B,4>, Tensor<B,4>, Tensor<B,4>, Tensor<B,4>, Tensor<B,4>, Tensor<B,4>) {
        let e1 = leaky_relu(self.e1_conv.forward(x));
        let e2 = leaky_relu(self.e2_gn.forward(self.e2_conv.forward(e1.clone())));
        let e3 = leaky_relu(self.e3_gn.forward(self.e3_conv.forward(e2.clone())));
        let e4 = leaky_relu(self.e4_gn.forward(self.e4_conv.forward(e3.clone())));
        let e5 = leaky_relu(self.e5_gn.forward(self.e5_conv.forward(e4.clone())));
        let e6 = leaky_relu(self.e6_gn.forward(self.e6_conv.forward(e5.clone())));
        let e7 = leaky_relu(self.e7_gn.forward(self.e7_conv.forward(e6.clone())));
        let e7 = self.bottleneck_attn(e7);
        let e7 = self.film_condition(e7, shape_cls, margin_cls);
        (e1, e2, e3, e4, e5, e6, e7)
    }

    fn decode(
        &self,
        e1: Tensor<B,4>, e2: Tensor<B,4>, e3: Tensor<B,4>,
        e4: Tensor<B,4>, e5: Tensor<B,4>, e6: Tensor<B,4>, e7: Tensor<B,4>,
    ) -> Tensor<B, 4> {
        // d7: expand bottleneck, no skip
        let d7 = self.drop.forward(activation::relu(self.d7_gn.forward(
            self.d7_conv.forward(upsample_nearest2x(e7))
        )));
        // d6: cat with e6 skip, upsample, conv
        let d6 = self.drop.forward(activation::relu(self.d6_gn.forward(
            self.d6_conv.forward(upsample_nearest2x(Tensor::cat(vec![d7, e6], 1)))
        )));
        // d5: cat with e5 skip, upsample, conv
        let d5 = self.drop.forward(activation::relu(self.d5_gn.forward(
            self.d5_conv.forward(upsample_nearest2x(Tensor::cat(vec![d6, e5], 1)))
        )));
        // d4: cat with e4 skip, upsample, conv
        let d4 = activation::relu(self.d4_gn.forward(
            self.d4_conv.forward(upsample_nearest2x(Tensor::cat(vec![d5, e4], 1)))
        ));
        // d3: cat with e3 skip, upsample, conv
        let d3 = activation::relu(self.d3_gn.forward(
            self.d3_conv.forward(upsample_nearest2x(Tensor::cat(vec![d4, e3], 1)))
        ));
        // d2: cat with e2 skip, upsample, conv
        let d2 = activation::relu(self.d2_gn.forward(
            self.d2_conv.forward(upsample_nearest2x(Tensor::cat(vec![d3, e2], 1)))
        ));
        // d1: cat with e1 skip, upsample, conv → logits
        let d1 = self.d1_conv.forward(upsample_nearest2x(Tensor::cat(vec![d2, e1], 1)));
        self.out_conv.forward(d1)
    }

    /// Returns raw logits [B, 1, H, W].
    pub fn forward(&self, x: Tensor<B, 4>, shape_cls: u32, margin_cls: u32) -> Tensor<B, 4> {
        let (e1, e2, e3, e4, e5, e6, e7) = self.encode(x, shape_cls, margin_cls);
        self.decode(e1, e2, e3, e4, e5, e6, e7)
    }

    /// Returns (logits [B,1,H,W], area_pred [B,1]) for training with area loss.
    pub fn forward_with_area(
        &self,
        x:          Tensor<B, 4>,
        shape_cls:  u32,
        margin_cls: u32,
    ) -> (Tensor<B, 4>, Tensor<B, 2>) {
        let (e1, e2, e3, e4, e5, e6, e7) = self.encode(x, shape_cls, margin_cls);
        // Area head: global average pool of bottleneck
        let [b, c, h7, w7] = e7.dims();
        let gap  = e7.clone().reshape([b, c, h7 * w7]).mean_dim(2).reshape([b, c]);
        let h1   = activation::relu(self.area_fc1.forward(gap));
        let area = activation::sigmoid(self.area_fc2.forward(h1));
        let logits = self.decode(e1, e2, e3, e4, e5, e6, e7);
        (logits, area)
    }

    /// Returns sigmoid probabilities [B, 1, H, W].
    pub fn forward_probs(&self, x: Tensor<B, 4>, shape_cls: u32, margin_cls: u32) -> Tensor<B, 4> {
        activation::sigmoid(self.forward(x, shape_cls, margin_cls))
    }
}

// ── PatchGAN Discriminator ────────────────────────────────────────────────────
//
// Input:  concat(cond [B,5,H,W], target [B,1,H,W]) → [B, 6, H, W]
// Output: [B, 1, patch_h, patch_w]  un-activated patch scores
// GroupNorm(8) used instead of BatchNorm — stable at any batch size (1–8),
// no running-stat accumulation issues when D receives one real + one fake pair.

#[derive(Module, Debug)]
pub struct PatchDiscriminator<B: Backend> {
    c1: Conv2d<B>,
    c2: Conv2d<B>, n2: GroupNorm<B>,
    c3: Conv2d<B>, n3: GroupNorm<B>,
    c4: Conv2d<B>, n4: GroupNorm<B>,
    c5: Conv2d<B>,
}

impl<B: Backend> PatchDiscriminator<B> {
    pub fn init(in_cond_channels: usize, device: &B::Device) -> Self {
        let total_in = in_cond_channels + 1;
        let s2 = |ic, oc| Conv2dConfig::new([ic, oc], [4, 4])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false);
        let s1 = |ic, oc| Conv2dConfig::new([ic, oc], [4, 4])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false);
        let gn = |ch| GroupNormConfig::new(8, ch);

        Self {
            c1: s2(total_in,  32).with_bias(true).init(device),
            c2: s2(32,   64).init(device), n2: gn(64).init(device),
            c3: s2(64,  128).init(device), n3: gn(128).init(device),
            c4: s1(128, 256).init(device), n4: gn(256).init(device),
            c5: s1(256,   1).with_bias(true).init(device),
        }
    }

    pub fn gpu_fence(&self) -> Tensor<B, 1> {
        self.c1.weight.val().flatten::<1>(0, 3).mean()
    }

    pub fn forward(&self, cond: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = Tensor::cat(vec![cond, target], 1);
        let x = leaky_relu(self.c1.forward(x));
        let x = leaky_relu(self.n2.forward(self.c2.forward(x)));
        let x = leaky_relu(self.n3.forward(self.c3.forward(x)));
        let x = leaky_relu(self.n4.forward(self.c4.forward(x)));
        self.c5.forward(x)
    }
}
