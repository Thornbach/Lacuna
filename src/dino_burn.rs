//! Hand-written BURN reimplementation of the DINOv3 ViT-B/16 multi-layer feature
//! extractor — the pure-Rust replacement for the `ort`/ONNX DINO path, so the app
//! can run the backbone on any burn backend (cuda / wgpu / ndarray) with NO
//! onnxruntime + cuDNN DLLs to ship.
//!
//! Architecture (verified against the HF `dinov3_vit` source + the exported ONNX;
//! see port/dino_spec + port/dino_reference.py):
//!   * ImageNet-normalize → patch conv (768×3×16×16, stride 16) → flatten row-major
//!   * prepend CLS + 4 register tokens  ⇒ 1 + 4 + g·g tokens
//!   * 12 pre-norm blocks: x += ls1·attn_rope(LN1 x); x += ls2·mlp_gelu(LN2 x)
//!       - attention: q/v/o have bias, k none; 2D-axial RoPE (θ=100) on patch tokens
//!       - LayerScale (lambda1); exact-erf GELU MLP (768→3072→768)
//!   * multi-layer head: hidden states after block 12 (-1) and block 7 (-6),
//!     last g·g patch tokens, L2-normalize each, concat ⇒ [B, g·g, 1536]
//!
//! Weights come from port/dino_weights.safetensors (HF state_dict, clean names).
//! Frozen inference only — no `#[derive(Module)]`, just loaded tensors + functional
//! ops, so there is zero Recorder/burn-import machinery.
//!
//! Batch-generic: `forward`/`forward_debug` take `[B,3,H,W]` for any B ≥ 1 —
//! every op either reads the batch dim from the input directly or broadcasts
//! against it (the CLS/register tokens are the one exception, fixed learned
//! `[1,·,768]` parameters repeated to B before concatenation). Lets a caller
//! batch many region crops into ONE forward pass instead of one call each —
//! the dominant per-region cost is the forward pass's fixed overhead, not
//! image size, since every crop is resized to the same `res`×`res` first.

use burn::tensor::{activation, backend::Backend, Tensor, TensorData};
use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;
use safetensors::SafeTensors;
use std::f32::consts::PI;

// ── Fixed ViT-B/16 dimensions ──────────────────────────────────────────────────
const DIM: usize = 768; // hidden size
const HEADS: usize = 12;
const HEAD_DIM: usize = DIM / HEADS; // 64
const INTER: usize = 3072; // MLP hidden
const N_BLOCKS: usize = 12;
const PATCH: usize = 16;
const N_REG: usize = 4;
const N_PREFIX: usize = 1 + N_REG; // cls + registers = 5
const LN_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 100.0;
// ImageNet normalization baked into the ONNX (verified: matches `mean`/`std` inits).
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
// Feature layers = (-1, -6): after block 11 (last) and after block 6 → concat order [blk11, blk6].
const FEAT_LAYER_A: usize = 11; // 0-based block index for hidden_states[-1]
const FEAT_LAYER_B: usize = 6; //  0-based block index for hidden_states[-6]

// ── Per-block weights ───────────────────────────────────────────────────────────
struct Block<B: Backend> {
    n1_w: Tensor<B, 1>,
    n1_b: Tensor<B, 1>,
    q_w: Tensor<B, 2>,
    q_b: Tensor<B, 1>,
    k_w: Tensor<B, 2>,
    v_w: Tensor<B, 2>,
    v_b: Tensor<B, 1>,
    o_w: Tensor<B, 2>,
    o_b: Tensor<B, 1>,
    ls1: Tensor<B, 1>,
    n2_w: Tensor<B, 1>,
    n2_b: Tensor<B, 1>,
    up_w: Tensor<B, 2>,
    up_b: Tensor<B, 1>,
    down_w: Tensor<B, 2>,
    down_b: Tensor<B, 1>,
    ls2: Tensor<B, 1>,
}

/// Loaded DINOv3 backbone. Generic over backend; construct once, reuse per tile.
pub struct DinoV3Burn<B: Backend> {
    device: B::Device,
    patch_w: Tensor<B, 4>,
    patch_b: Tensor<B, 1>,
    cls: Tensor<B, 3>,       // [1,1,768]
    registers: Tensor<B, 3>, // [1,4,768]
    blocks: Vec<Block<B>>,
}

/// Intermediate activations for stage-by-stage validation against the golden ref.
pub struct DinoDebug<B: Backend> {
    pub embed: Tensor<B, 3>,   // after patch-embed + prepend  (== HF hidden_states[0])
    pub block0: Tensor<B, 3>,  // after block 0
    pub block6: Tensor<B, 3>,  // after block 6  (hidden_states[7])
    pub block11: Tensor<B, 3>, // after block 11 (hidden_states[12])
    pub feat: Tensor<B, 3>,    // final multilayer [B, g·g, 1536]
}

// ── weight loading ──────────────────────────────────────────────────────────────
fn load<B: Backend, const D: usize>(
    st: &SafeTensors,
    name: &str,
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D> {
    let v = st
        .tensor(name)
        .unwrap_or_else(|_| panic!("missing tensor `{name}` in safetensors"));
    let bytes = v.data();
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let n: usize = shape.iter().product();
    assert_eq!(
        data.len(),
        n,
        "`{name}` element count {} != expected {:?}",
        data.len(),
        shape
    );
    Tensor::from_data(TensorData::new(data, shape), device)
}

impl<B: Backend> DinoV3Burn<B> {
    /// Load from an HF-state_dict safetensors (port/dino_weights.safetensors).
    pub fn load(path: &str, device: &B::Device) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        let st = SafeTensors::deserialize(&bytes).map_err(|e| format!("parse {path}: {e}"))?;

        let patch_w = load(&st, "embeddings.patch_embeddings.weight", [DIM, 3, PATCH, PATCH], device);
        let patch_b = load(&st, "embeddings.patch_embeddings.bias", [DIM], device);
        let cls = load(&st, "embeddings.cls_token", [1, 1, DIM], device);
        let registers = load(&st, "embeddings.register_tokens", [1, N_REG, DIM], device);

        let mut blocks = Vec::with_capacity(N_BLOCKS);
        for i in 0..N_BLOCKS {
            let p = |s: &str| format!("model.layer.{i}.{s}");
            blocks.push(Block {
                n1_w: load(&st, &p("norm1.weight"), [DIM], device),
                n1_b: load(&st, &p("norm1.bias"), [DIM], device),
                q_w: load(&st, &p("attention.q_proj.weight"), [DIM, DIM], device),
                q_b: load(&st, &p("attention.q_proj.bias"), [DIM], device),
                k_w: load(&st, &p("attention.k_proj.weight"), [DIM, DIM], device),
                v_w: load(&st, &p("attention.v_proj.weight"), [DIM, DIM], device),
                v_b: load(&st, &p("attention.v_proj.bias"), [DIM], device),
                o_w: load(&st, &p("attention.o_proj.weight"), [DIM, DIM], device),
                o_b: load(&st, &p("attention.o_proj.bias"), [DIM], device),
                ls1: load(&st, &p("layer_scale1.lambda1"), [DIM], device),
                n2_w: load(&st, &p("norm2.weight"), [DIM], device),
                n2_b: load(&st, &p("norm2.bias"), [DIM], device),
                up_w: load(&st, &p("mlp.up_proj.weight"), [INTER, DIM], device),
                up_b: load(&st, &p("mlp.up_proj.bias"), [INTER], device),
                down_w: load(&st, &p("mlp.down_proj.weight"), [DIM, INTER], device),
                down_b: load(&st, &p("mlp.down_proj.bias"), [DIM], device),
                ls2: load(&st, &p("layer_scale2.lambda1"), [DIM], device),
            });
        }
        Ok(Self { device: device.clone(), patch_w, patch_b, cls, registers, blocks })
    }

    /// Full forward: input `[B,3,H,W]` in [0,1] (raw RGB) → `[B, g·g, 1536]`.
    /// `B` (batch) is read from `x` itself — every image in the batch shares
    /// the same `H,W` (the caller resizes each crop to the same `res` first,
    /// same as the single-image path always did per-call), so one forward
    /// pass processes the whole batch instead of one call per image.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        self.forward_debug(x).feat
    }

    /// Forward keeping the intermediate activations used for validation.
    pub fn forward_debug(&self, x: Tensor<B, 4>) -> DinoDebug<B> {
        let dev = &self.device;
        let [batch, _, h, w] = x.dims();
        let gh = h / PATCH;
        let gw = w / PATCH;
        let npatch = gh * gw;
        let ntok = N_PREFIX + npatch;

        // ImageNet normalize.
        let mean = Tensor::<B, 1>::from_data(TensorData::new(MEAN.to_vec(), [3]), dev).reshape([1, 3, 1, 1]);
        let std = Tensor::<B, 1>::from_data(TensorData::new(STD.to_vec(), [3]), dev).reshape([1, 3, 1, 1]);
        let x = (x - mean) / std;

        // Patch embed: conv2d stride 16 → [B,768,gh,gw] → [B, npatch, 768].
        let opt = ConvOptions::new([PATCH, PATCH], [0, 0], [1, 1], 1);
        let patches = conv2d(x, self.patch_w.clone(), Some(self.patch_b.clone()), opt); // [B,768,gh,gw]
        let patches = patches.reshape([batch, DIM, npatch]).swap_dims(1, 2); // [B, npatch, 768]

        // Prepend CLS + register tokens — loaded as fixed [1,·,768] learned
        // parameters, so they need repeating to the batch's actual size
        // before they can be concatenated with `patches` (skip the repeat
        // at batch=1, the common single-region case, to avoid a pointless copy).
        let (cls, registers) = if batch == 1 {
            (self.cls.clone(), self.registers.clone())
        } else {
            (self.cls.clone().repeat_dim(0, batch), self.registers.clone().repeat_dim(0, batch))
        };
        let emb = Tensor::cat(vec![cls, registers, patches], 1); // [B, ntok, 768]

        // RoPE tables for the patch tokens only.
        let (cos, sin) = rope_tables::<B>(gh, gw, dev); // [npatch, 64]

        let mut x = emb.clone();
        let mut block0 = None;
        let mut block6 = None;
        let mut block11 = None;
        for (i, blk) in self.blocks.iter().enumerate() {
            x = block_forward(blk, x, &cos, &sin, ntok, npatch);
            match i {
                0 => block0 = Some(x.clone()),
                FEAT_LAYER_B => block6 = Some(x.clone()),
                FEAT_LAYER_A => block11 = Some(x.clone()),
                _ => {}
            }
        }
        let block6 = block6.unwrap();
        let block11 = block11.unwrap();

        // Multi-layer head: last npatch tokens of blk11 & blk6, L2-norm, concat.
        let a = l2norm(patch_tokens(block11.clone(), ntok, npatch));
        let b = l2norm(patch_tokens(block6.clone(), ntok, npatch));
        let feat = Tensor::cat(vec![a, b], 2); // [B, npatch, 1536]

        DinoDebug {
            embed: emb,
            block0: block0.unwrap(),
            block6,
            block11,
            feat,
        }
    }
}

// ── functional helpers ──────────────────────────────────────────────────────────

/// y = x · Wᵀ + b   (x: [1,N,In], W: [Out,In]) → [1,N,Out].
fn linear<B: Backend>(x: Tensor<B, 3>, w: &Tensor<B, 2>, b: Option<&Tensor<B, 1>>) -> Tensor<B, 3> {
    let [inn, out] = [w.dims()[1], w.dims()[0]];
    let wt = w.clone().swap_dims(0, 1).reshape([1, inn, out]); // [1,In,Out]
    let mut y = x.matmul(wt);
    if let Some(bb) = b {
        y = y + bb.clone().reshape([1, 1, out]);
    }
    y
}

/// LayerNorm over the last dim (biased variance, like torch).
fn layer_norm<B: Backend>(x: Tensor<B, 3>, w: &Tensor<B, 1>, b: &Tensor<B, 1>) -> Tensor<B, 3> {
    let d = x.dims()[2];
    let mean = x.clone().mean_dim(2); // [1,N,1]
    let xc = x - mean;
    let var = xc.clone().powf_scalar(2.0).mean_dim(2); // [1,N,1]
    let xn = xc / var.add_scalar(LN_EPS).sqrt();
    xn * w.clone().reshape([1, 1, d]) + b.clone().reshape([1, 1, d])
}

/// L2-normalize over the last dim (eps 1e-12, like F.normalize).
fn l2norm<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let n = x.clone().powf_scalar(2.0).sum_dim(2).sqrt(); // [1,N,1]
    x / n.clamp_min(1e-12)
}

/// Last `npatch` tokens (drop the prefix cls+registers).
fn patch_tokens<B: Backend>(x: Tensor<B, 3>, ntok: usize, npatch: usize) -> Tensor<B, 3> {
    let batch = x.dims()[0];
    x.slice([0..batch, (ntok - npatch)..ntok, 0..DIM])
}

/// rotate_half over the last dim of a [1,H,N,64] tensor: [-x2, x1].
fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, h, n, d] = x.dims();
    let half = d / 2;
    let x1 = x.clone().slice([0..b, 0..h, 0..n, 0..half]);
    let x2 = x.slice([0..b, 0..h, 0..n, half..d]);
    Tensor::cat(vec![-x2, x1], 3)
}

/// Apply RoPE to the patch tokens of q/k: [1,H,ntok,64], cos/sin [npatch,64].
fn apply_rope<B: Backend>(
    x: Tensor<B, 4>,
    cos: &Tensor<B, 2>,
    sin: &Tensor<B, 2>,
    ntok: usize,
    npatch: usize,
) -> Tensor<B, 4> {
    let [b, h, _, d] = x.dims();
    let start = ntok - npatch;
    let prefix = x.clone().slice([0..b, 0..h, 0..start, 0..d]);
    let pat = x.slice([0..b, 0..h, start..ntok, 0..d]); // [1,H,npatch,64]
    let cos4 = cos.clone().reshape([1, 1, npatch, d]);
    let sin4 = sin.clone().reshape([1, 1, npatch, d]);
    let rotated = pat.clone() * cos4 + rotate_half(pat) * sin4;
    Tensor::cat(vec![prefix, rotated], 2)
}

fn block_forward<B: Backend>(
    blk: &Block<B>,
    x: Tensor<B, 3>,
    cos: &Tensor<B, 2>,
    sin: &Tensor<B, 2>,
    ntok: usize,
    npatch: usize,
) -> Tensor<B, 3> {
    let batch = x.dims()[0];
    // ── attention ──
    let h = layer_norm(x.clone(), &blk.n1_w, &blk.n1_b);
    let q = linear(h.clone(), &blk.q_w, Some(&blk.q_b));
    let k = linear(h.clone(), &blk.k_w, None);
    let v = linear(h, &blk.v_w, Some(&blk.v_b));

    // [B,ntok,768] → [B,H,ntok,64]
    let to_heads = |t: Tensor<B, 3>| t.reshape([batch, ntok, HEADS, HEAD_DIM]).swap_dims(1, 2);
    let q = apply_rope(to_heads(q), cos, sin, ntok, npatch);
    let k = apply_rope(to_heads(k), cos, sin, ntok, npatch);
    let v = to_heads(v);

    let scale = (HEAD_DIM as f32).powf(-0.5);
    let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale); // [B,H,ntok,ntok]
    let attn = activation::softmax(scores, 3);
    let ctx = attn.matmul(v); // [B,H,ntok,64]
    let ctx = ctx.swap_dims(1, 2).reshape([batch, ntok, DIM]); // merge heads
    let attn_out = linear(ctx, &blk.o_w, Some(&blk.o_b));
    let x = x + attn_out * blk.ls1.clone().reshape([1, 1, DIM]);

    // ── MLP ──
    let h = layer_norm(x.clone(), &blk.n2_w, &blk.n2_b);
    let h = linear(h, &blk.up_w, Some(&blk.up_b));
    let h = activation::gelu(h); // exact erf GELU
    let h = linear(h, &blk.down_w, Some(&blk.down_b));
    x + h * blk.ls2.clone().reshape([1, 1, DIM])
}

/// 2D-axial RoPE cos/sin for a gh×gw patch grid → [gh·gw, HEAD_DIM].
/// angles[p] = [y·f (16), x·f (16), y·f (16), x·f (16)] with f = 2π·inv_freq,
/// inv_freq[i] = θ^(−4i/HEAD_DIM), coords = 2·((idx+0.5)/g) − 1  (matches HF).
fn rope_tables<B: Backend>(gh: usize, gw: usize, device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let nf = HEAD_DIM / 4; // 16
    let inv_freq: Vec<f32> = (0..nf)
        .map(|i| ROPE_THETA.powf(-(4.0 * i as f32) / HEAD_DIM as f32))
        .collect();
    let npatch = gh * gw;
    let mut ang = vec![0f32; npatch * HEAD_DIM];
    for hh in 0..gh {
        let yc = 2.0 * ((hh as f32 + 0.5) / gh as f32) - 1.0;
        for ww in 0..gw {
            let xc = 2.0 * ((ww as f32 + 0.5) / gw as f32) - 1.0;
            let p = hh * gw + ww;
            for i in 0..nf {
                let ay = 2.0 * PI * yc * inv_freq[i];
                let ax = 2.0 * PI * xc * inv_freq[i];
                ang[p * HEAD_DIM + i] = ay;
                ang[p * HEAD_DIM + nf + i] = ax;
                ang[p * HEAD_DIM + 2 * nf + i] = ay;
                ang[p * HEAD_DIM + 3 * nf + i] = ax;
            }
        }
    }
    let a = Tensor::<B, 2>::from_data(TensorData::new(ang, [npatch, HEAD_DIM]), device);
    (a.clone().cos(), a.sin())
}
