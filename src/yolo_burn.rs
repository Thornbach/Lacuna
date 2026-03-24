//! Hand-written BURN reimplementation of the fused YOLO26m-seg network — the
//! pure-Rust replacement for the `ort`/ONNX YOLO path. Backend-generic, frozen
//! inference (loaded tensors + functional ops, no #[derive(Module)]).
//!
//! Produces the RAW pre-decode head outputs; the anchor decode + top-k selection
//! are done in Rust (`yolo_decode`), matching ultralytics' end2end path exactly.
//!
//! Architecture (from the fused ultralytics model + weight map, see port/):
//!   backbone: Conv,Conv, C3k2, Conv, C3k2, Conv, C3k2, Conv, C3k2, SPPF, C2PSA
//!   neck(PAN): Up,Cat,C3k2, Up,Cat,C3k2, Conv,Cat,C3k2, Conv,Cat,C3k2
//!   head Segment26: one2one box/cls/mask branches (P3,P4,P5) + Proto26
//! All Conv = fused conv2d(+bias)+SiLU; reg_max=1 (no DFL); nc=1; nm=32.

use burn::tensor::module::{conv2d, conv_transpose2d, max_pool2d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn::tensor::{activation, backend::Backend, Tensor, TensorData};
use safetensors::SafeTensors;

// ── generic fused conv (conv2d + optional SiLU) ─────────────────────────────────
struct Cv<B: Backend> {
    w: Tensor<B, 4>,
    b: Tensor<B, 1>,
    stride: usize,
    pad: usize,
    groups: usize,
    act: bool,
}
impl<B: Backend> Cv<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let opt = ConvOptions::new([self.stride, self.stride], [self.pad, self.pad], [1, 1], self.groups);
        let y = conv2d(x, self.w.clone(), Some(self.b.clone()), opt);
        if self.act {
            activation::silu(y)
        } else {
            y
        }
    }
}

fn ld<B: Backend, const D: usize>(st: &SafeTensors, name: &str, dev: &B::Device) -> Tensor<B, D> {
    let v = st.tensor(name).unwrap_or_else(|_| panic!("missing tensor `{name}`"));
    let shp: [usize; D] = v
        .shape()
        .try_into()
        .unwrap_or_else(|_| panic!("`{name}` rank {} != {}", v.shape().len(), D));
    let data: Vec<f32> = v
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Tensor::from_data(TensorData::new(data, shp), dev)
}

fn cv_named<B: Backend>(st: &SafeTensors, wn: &str, bn: &str, stride: usize, act: bool, dev: &B::Device) -> Cv<B> {
    let w: Tensor<B, 4> = ld(st, wn, dev);
    let b: Tensor<B, 1> = ld(st, bn, dev);
    let [outc, inpg, k, _] = w.dims();
    let groups = if inpg == 1 && outc > 1 { outc } else { 1 };
    Cv { w, b, stride, pad: k / 2, groups, act }
}
/// Conv block: `{p}.conv.weight/bias`, SiLU per `act`.
fn cv<B: Backend>(st: &SafeTensors, p: &str, stride: usize, act: bool, dev: &B::Device) -> Cv<B> {
    cv_named(st, &format!("{p}.conv.weight"), &format!("{p}.conv.bias"), stride, act, dev)
}
/// Raw nn.Conv2d (head final layers): `{p}.weight/bias`, no act.
fn cv_plain<B: Backend>(st: &SafeTensors, p: &str, dev: &B::Device) -> Cv<B> {
    cv_named(st, &format!("{p}.weight"), &format!("{p}.bias"), 1, false, dev)
}

// ── building blocks ─────────────────────────────────────────────────────────────
struct Bottleneck<B: Backend> {
    cv1: Cv<B>,
    cv2: Cv<B>,
    add: bool,
}
impl<B: Backend> Bottleneck<B> {
    fn load(st: &SafeTensors, p: &str, add: bool, dev: &B::Device) -> Self {
        Self { cv1: cv(st, &format!("{p}.cv1"), 1, true, dev), cv2: cv(st, &format!("{p}.cv2"), 1, true, dev), add }
    }
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.cv2.forward(self.cv1.forward(x.clone()));
        if self.add {
            x + y
        } else {
            y
        }
    }
}

struct C3k<B: Backend> {
    cv1: Cv<B>,
    cv2: Cv<B>,
    cv3: Cv<B>,
    m: Vec<Bottleneck<B>>,
}
impl<B: Backend> C3k<B> {
    fn load(st: &SafeTensors, p: &str, n: usize, dev: &B::Device) -> Self {
        let m = (0..n).map(|j| Bottleneck::load(st, &format!("{p}.m.{j}"), true, dev)).collect();
        Self {
            cv1: cv(st, &format!("{p}.cv1"), 1, true, dev),
            cv2: cv(st, &format!("{p}.cv2"), 1, true, dev),
            cv3: cv(st, &format!("{p}.cv3"), 1, true, dev),
            m,
        }
    }
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut a = self.cv1.forward(x.clone());
        for b in &self.m {
            a = b.forward(a);
        }
        let b = self.cv2.forward(x);
        self.cv3.forward(Tensor::cat(vec![a, b], 1))
    }
}

// ── attention (C2PSA / layer-22 PSABlock) ───────────────────────────────────────
struct Attention<B: Backend> {
    qkv: Cv<B>,
    proj: Cv<B>,
    pe: Cv<B>,
    num_heads: usize,
    key_dim: usize,
    head_dim: usize,
}
impl<B: Backend> Attention<B> {
    fn load(st: &SafeTensors, p: &str, dev: &B::Device) -> Self {
        let qkv = cv(st, &format!("{p}.qkv"), 1, false, dev);
        let proj = cv(st, &format!("{p}.proj"), 1, false, dev);
        let pe = cv(st, &format!("{p}.pe"), 1, false, dev);
        let dim = proj.w.dims()[0]; // 256
        let num_heads = dim / 64;
        let head_dim = dim / num_heads; // 64
        let key_dim = head_dim / 2; // 32
        Self { qkv, proj, pe, num_heads, key_dim, head_dim }
    }
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();
        let n = h * w;
        let nh = self.num_heads;
        let kd = self.key_dim;
        let hd = self.head_dim;
        let qkv = self.qkv.forward(x); // [1, 2c, h, w]
        let qkv = qkv.reshape([b, nh, kd * 2 + hd, n]); // [1, nh, 128, N]
        let q = qkv.clone().slice([0..b, 0..nh, 0..kd, 0..n]); // [1,nh,32,N]
        let k = qkv.clone().slice([0..b, 0..nh, kd..(2 * kd), 0..n]); // [1,nh,32,N]
        let v = qkv.slice([0..b, 0..nh, (2 * kd)..(2 * kd + hd), 0..n]); // [1,nh,64,N]

        let scale = (kd as f32).powf(-0.5);
        let attn = q.swap_dims(2, 3).matmul(k).mul_scalar(scale); // [1,nh,N,N]
        let attn = activation::softmax(attn, 3);
        let ctx = v.clone().matmul(attn.swap_dims(2, 3)); // [1,nh,64,N]
        let ctx = ctx.reshape([b, c, h, w]);
        let pe = self.pe.forward(v.reshape([b, c, h, w]));
        self.proj.forward(ctx + pe)
    }
}

struct PSABlock<B: Backend> {
    attn: Attention<B>,
    ffn0: Cv<B>,
    ffn1: Cv<B>,
}
impl<B: Backend> PSABlock<B> {
    fn load(st: &SafeTensors, p: &str, dev: &B::Device) -> Self {
        Self {
            attn: Attention::load(st, &format!("{p}.attn"), dev),
            ffn0: cv(st, &format!("{p}.ffn.0"), 1, true, dev),
            ffn1: cv(st, &format!("{p}.ffn.1"), 1, false, dev),
        }
    }
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = x.clone() + self.attn.forward(x);
        let f = self.ffn1.forward(self.ffn0.forward(x.clone()));
        x + f
    }
}

// ── C3k2 (C2f-style) with a single block: C3k, or (layer22) Seq(Bottleneck,PSABlock) ──
enum C3k2Block<B: Backend> {
    C3k(C3k<B>),
    BottPsa(Bottleneck<B>, PSABlock<B>),
}
impl<B: Backend> C3k2Block<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            C3k2Block::C3k(m) => m.forward(x),
            C3k2Block::BottPsa(b, p) => p.forward(b.forward(x)),
        }
    }
}
struct C3k2<B: Backend> {
    cv1: Cv<B>,
    cv2: Cv<B>,
    block: C3k2Block<B>,
}
impl<B: Backend> C3k2<B> {
    /// `attn_variant` = layer-22 Seq(Bottleneck, PSABlock); else C3k(inner=2).
    fn load(st: &SafeTensors, p: &str, attn_variant: bool, dev: &B::Device) -> Self {
        let block = if attn_variant {
            C3k2Block::BottPsa(
                Bottleneck::load(st, &format!("{p}.m.0.0"), true, dev),
                PSABlock::load(st, &format!("{p}.m.0.1"), dev),
            )
        } else {
            C3k2Block::C3k(C3k::load(st, &format!("{p}.m.0"), 2, dev))
        };
        Self { cv1: cv(st, &format!("{p}.cv1"), 1, true, dev), cv2: cv(st, &format!("{p}.cv2"), 1, true, dev), block }
    }
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let t = self.cv1.forward(x);
        let parts = t.chunk(2, 1); // [a, b]
        let a = parts[0].clone();
        let b = parts[1].clone();
        let c = self.block.forward(b.clone());
        self.cv2.forward(Tensor::cat(vec![a, b, c], 1))
    }
}

// ── SPPF ────────────────────────────────────────────────────────────────────────
struct Sppf<B: Backend> {
    cv1: Cv<B>,
    cv2: Cv<B>,
    add: bool,
}
impl<B: Backend> Sppf<B> {
    fn load(st: &SafeTensors, p: &str, add: bool, dev: &B::Device) -> Self {
        // NOTE: SPPF cv1 has act=False (no SiLU); cv2 has default SiLU.
        Self { cv1: cv(st, &format!("{p}.cv1"), 1, false, dev), cv2: cv(st, &format!("{p}.cv2"), 1, true, dev), add }
    }
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y0 = self.cv1.forward(x.clone());
        let y1 = max_pool2d(y0.clone(), [5, 5], [1, 1], [2, 2], [1, 1]);
        let y2 = max_pool2d(y1.clone(), [5, 5], [1, 1], [2, 2], [1, 1]);
        let y3 = max_pool2d(y2.clone(), [5, 5], [1, 1], [2, 2], [1, 1]);
        let out = self.cv2.forward(Tensor::cat(vec![y0, y1, y2, y3], 1));
        if self.add {
            out + x
        } else {
            out
        }
    }
}

struct C2psa<B: Backend> {
    cv1: Cv<B>,
    cv2: Cv<B>,
    m: Vec<PSABlock<B>>,
}
impl<B: Backend> C2psa<B> {
    fn load(st: &SafeTensors, p: &str, n: usize, dev: &B::Device) -> Self {
        let m = (0..n).map(|j| PSABlock::load(st, &format!("{p}.m.{j}"), dev)).collect();
        Self { cv1: cv(st, &format!("{p}.cv1"), 1, true, dev), cv2: cv(st, &format!("{p}.cv2"), 1, true, dev), m }
    }
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let t = self.cv1.forward(x);
        let parts = t.chunk(2, 1);
        let a = parts[0].clone();
        let mut b = parts[1].clone();
        for blk in &self.m {
            b = blk.forward(b);
        }
        self.cv2.forward(Tensor::cat(vec![a, b], 1))
    }
}

// ── Proto26 ─────────────────────────────────────────────────────────────────────
struct Proto26<B: Backend> {
    refine: Vec<Cv<B>>, // 2 (for P4, P5)
    fuse: Cv<B>,
    cv1: Cv<B>,
    up_w: Tensor<B, 4>,
    up_b: Tensor<B, 1>,
    cv2: Cv<B>,
    cv3: Cv<B>,
}
impl<B: Backend> Proto26<B> {
    fn load(st: &SafeTensors, p: &str, dev: &B::Device) -> Self {
        Self {
            refine: vec![
                cv(st, &format!("{p}.feat_refine.0"), 1, true, dev),
                cv(st, &format!("{p}.feat_refine.1"), 1, true, dev),
            ],
            fuse: cv(st, &format!("{p}.feat_fuse"), 1, true, dev),
            cv1: cv(st, &format!("{p}.cv1"), 1, true, dev),
            up_w: ld(st, &format!("{p}.upsample.weight"), dev),
            up_b: ld(st, &format!("{p}.upsample.bias"), dev),
            cv2: cv(st, &format!("{p}.cv2"), 1, true, dev),
            cv3: cv(st, &format!("{p}.cv3"), 1, true, dev),
        }
    }
    fn forward(&self, feats: [Tensor<B, 4>; 3]) -> Tensor<B, 4> {
        let [p3, p4, p5] = feats;
        let [_, _, th, tw] = p3.dims();
        let mut feat = p3;
        for (i, src) in [p4, p5].into_iter().enumerate() {
            let up = nearest_up_to(self.refine[i].forward(src), th, tw);
            feat = feat + up;
        }
        let t = self.cv1.forward(self.fuse.forward(feat));
        let opt = ConvTransposeOptions::new([2, 2], [0, 0], [0, 0], [1, 1], 1);
        let t = conv_transpose2d(t, self.up_w.clone(), Some(self.up_b.clone()), opt);
        self.cv3.forward(self.cv2.forward(t))
    }
}

// ── head branches ───────────────────────────────────────────────────────────────
struct Seq3<B: Backend> {
    a: Cv<B>,
    b: Cv<B>,
    out: Cv<B>,
} // box/mask: Conv,Conv,Conv2d
impl<B: Backend> Seq3<B> {
    fn load(st: &SafeTensors, p: &str, dev: &B::Device) -> Self {
        Self {
            a: cv(st, &format!("{p}.0"), 1, true, dev),
            b: cv(st, &format!("{p}.1"), 1, true, dev),
            out: cv_plain(st, &format!("{p}.2"), dev),
        }
    }
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.out.forward(self.b.forward(self.a.forward(x)))
    }
}
struct ClsBranch<B: Backend> {
    dw0: Cv<B>,
    pw0: Cv<B>,
    dw1: Cv<B>,
    pw1: Cv<B>,
    out: Cv<B>,
}
impl<B: Backend> ClsBranch<B> {
    fn load(st: &SafeTensors, p: &str, dev: &B::Device) -> Self {
        Self {
            dw0: cv(st, &format!("{p}.0.0"), 1, true, dev),
            pw0: cv(st, &format!("{p}.0.1"), 1, true, dev),
            dw1: cv(st, &format!("{p}.1.0"), 1, true, dev),
            pw1: cv(st, &format!("{p}.1.1"), 1, true, dev),
            out: cv_plain(st, &format!("{p}.2"), dev),
        }
    }
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.pw0.forward(self.dw0.forward(x));
        let x = self.pw1.forward(self.dw1.forward(x));
        self.out.forward(x)
    }
}

/// Nearest upsample of [b,c,h,w] to (th,tw) by an integer factor (th/h == tw/w).
fn nearest_up_to<B: Backend>(x: Tensor<B, 4>, th: usize, tw: usize) -> Tensor<B, 4> {
    let [b, c, h, w] = x.dims();
    let f = th / h;
    debug_assert_eq!(h * f, th);
    debug_assert_eq!(w * (tw / w), tw);
    x.reshape([b, c, h, 1, w, 1])
        .repeat_dim(3, f)
        .repeat_dim(5, tw / w)
        .reshape([b, c, th, tw])
}

/// Raw pre-decode outputs, plus validation checkpoints.
pub struct YoloRaw<B: Backend> {
    pub layer0: Tensor<B, 4>,
    pub layer4: Tensor<B, 4>,
    pub layer9: Tensor<B, 4>,
    pub layer10: Tensor<B, 4>,
    pub layer16: Tensor<B, 4>,
    pub layer19: Tensor<B, 4>,
    pub layer22: Tensor<B, 4>,
    pub proto: Tensor<B, 4>,   // [1,32,160,160]
    pub head_box: Tensor<B, 3>, // [1,4,8400]
    pub head_cls: Tensor<B, 3>, // [1,1,8400]
    pub head_msk: Tensor<B, 3>, // [1,32,8400]
}

pub struct YoloV26Burn<B: Backend> {
    device: B::Device,
    // backbone
    c0: Cv<B>,
    c1: Cv<B>,
    l2: C3k2<B>,
    c3: Cv<B>,
    l4: C3k2<B>,
    c5: Cv<B>,
    l6: C3k2<B>,
    c7: Cv<B>,
    l8: C3k2<B>,
    sppf: Sppf<B>,
    c2psa: C2psa<B>,
    // neck
    l13: C3k2<B>,
    l16: C3k2<B>,
    c17: Cv<B>,
    l19: C3k2<B>,
    c20: Cv<B>,
    l22: C3k2<B>,
    // head
    box_br: Vec<Seq3<B>>,
    cls_br: Vec<ClsBranch<B>>,
    msk_br: Vec<Seq3<B>>,
    proto: Proto26<B>,
}

impl<B: Backend> YoloV26Burn<B> {
    pub fn load(path: &str, device: &B::Device) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        let st = SafeTensors::deserialize(&bytes).map_err(|e| format!("parse {path}: {e}"))?;
        let m = |i: usize| format!("model.{i}");
        Ok(Self {
            device: device.clone(),
            c0: cv(&st, &m(0), 2, true, device),
            c1: cv(&st, &m(1), 2, true, device),
            l2: C3k2::load(&st, &m(2), false, device),
            c3: cv(&st, &m(3), 2, true, device),
            l4: C3k2::load(&st, &m(4), false, device),
            c5: cv(&st, &m(5), 2, true, device),
            l6: C3k2::load(&st, &m(6), false, device),
            c7: cv(&st, &m(7), 2, true, device),
            l8: C3k2::load(&st, &m(8), false, device),
            sppf: Sppf::load(&st, &m(9), true, device),
            c2psa: C2psa::load(&st, &m(10), 1, device),
            l13: C3k2::load(&st, &m(13), false, device),
            l16: C3k2::load(&st, &m(16), false, device),
            c17: cv(&st, &m(17), 2, true, device),
            l19: C3k2::load(&st, &m(19), false, device),
            c20: cv(&st, &m(20), 2, true, device),
            l22: C3k2::load(&st, &m(22), true, device), // attn variant
            box_br: (0..3).map(|s| Seq3::load(&st, &format!("model.23.one2one_cv2.{s}"), device)).collect(),
            cls_br: (0..3).map(|s| ClsBranch::load(&st, &format!("model.23.one2one_cv3.{s}"), device)).collect(),
            msk_br: (0..3).map(|s| Seq3::load(&st, &format!("model.23.one2one_cv4.{s}"), device)).collect(),
            proto: Proto26::load(&st, "model.23.proto", device),
        })
    }

    /// Input `[1,3,640,640]` in [0,1] → raw head outputs + proto (+ checkpoints).
    pub fn forward(&self, x: Tensor<B, 4>) -> YoloRaw<B> {
        let x0 = self.c0.forward(x);
        let x1 = self.c1.forward(x0.clone());
        let x2 = self.l2.forward(x1);
        let x3 = self.c3.forward(x2);
        let x4 = self.l4.forward(x3);
        let x4c = x4.clone(); // validation checkpoint
        let x5 = self.c5.forward(x4.clone());
        let x6 = self.l6.forward(x5);
        let x7 = self.c7.forward(x6.clone());
        let x8 = self.l8.forward(x7);
        let x9 = self.sppf.forward(x8);
        let x9c = x9.clone(); // validation checkpoint
        let x10 = self.c2psa.forward(x9);
        let x10c = x10.clone(); // validation checkpoint
        // neck
        let x11 = nearest_up_to(x10.clone(), x6.dims()[2], x6.dims()[3]);
        let x12 = Tensor::cat(vec![x11, x6.clone()], 1);
        let x13 = self.l13.forward(x12);
        let x14 = nearest_up_to(x13.clone(), x4.dims()[2], x4.dims()[3]);
        let x15 = Tensor::cat(vec![x14, x4], 1);
        let x16 = self.l16.forward(x15);
        let x17 = self.c17.forward(x16.clone());
        let x18 = Tensor::cat(vec![x17, x13], 1);
        let x19 = self.l19.forward(x18);
        let x20 = self.c20.forward(x19.clone());
        let x21 = Tensor::cat(vec![x20, x10], 1);
        let x22 = self.l22.forward(x21);

        let feats = [x16.clone(), x19.clone(), x22.clone()];
        let proto = self.proto.forward([x16.clone(), x19.clone(), x22.clone()]);

        // head branches per scale, concat over scales [P3,P4,P5] → [1,C,8400]
        let mut boxes = Vec::new();
        let mut clss = Vec::new();
        let mut msks = Vec::new();
        for (s, f) in feats.iter().enumerate() {
            let [_, _, h, w] = f.dims();
            let n = h * w;
            boxes.push(self.box_br[s].forward(f.clone()).reshape([1, 4, n]));
            clss.push(self.cls_br[s].forward(f.clone()).reshape([1, 1, n]));
            msks.push(self.msk_br[s].forward(f.clone()).reshape([1, 32, n]));
        }
        YoloRaw {
            layer0: x0,
            layer4: x4c,
            layer9: x9c,
            layer10: x10c,
            layer16: x16,
            layer19: x19,
            layer22: x22,
            proto,
            head_box: Tensor::cat(boxes, 2),
            head_cls: Tensor::cat(clss, 2),
            head_msk: Tensor::cat(msks, 2),
        }
    }
}

/// Rust decode of the raw head outputs → the SAME `(output0, output1)` the ort graph
/// emits: `d0` = 300×38 rows `[x1,y1,x2,y2, score, cls, c0..c31]` (letterbox-pixel
/// coords), `d1` = proto `[32·ph·pw]`. Reproduces ultralytics' end2end path:
/// anchors at cell centers (+0.5), dist2bbox (ltrb→xyxy) × stride, sigmoid cls, top-300.
/// Returns `(d0, d1, [ndet, nattr, nproto, ph, pw])`.
pub fn yolo_decode<B: Backend>(raw: YoloRaw<B>, imgsz: usize) -> (Vec<f32>, Vec<f32>, [usize; 5]) {
    const NM: usize = 32;
    const NATTR: usize = 4 + 1 + 1 + NM; // 38
    const NDET: usize = 300;
    let strides = [8usize, 16, 32];

    let boxes = raw.head_box.into_data().to_vec::<f32>().unwrap(); // [4*8400] row = channel-major
    let cls = raw.head_cls.into_data().to_vec::<f32>().unwrap(); // [1*8400]
    let coeff = raw.head_msk.into_data().to_vec::<f32>().unwrap(); // [32*8400]
    let [_, _, ph, pw] = raw.proto.dims();
    let d1 = raw.proto.into_data().to_vec::<f32>().unwrap();

    // Anchor grid (cell centers) + per-anchor stride, in the head's scale/row-major order.
    let na: usize = strides.iter().map(|&s| (imgsz / s) * (imgsz / s)).sum(); // 8400
    let mut ax = Vec::with_capacity(na);
    let mut ay = Vec::with_capacity(na);
    let mut ast = Vec::with_capacity(na);
    for &s in &strides {
        let g = imgsz / s;
        for y in 0..g {
            for x in 0..g {
                ax.push(x as f32 + 0.5);
                ay.push(y as f32 + 0.5);
                ast.push(s as f32);
            }
        }
    }

    // Decode every anchor, keep (score, index).
    let mut scored: Vec<(f32, usize)> = (0..na)
        .map(|a| (1.0 / (1.0 + (-cls[a]).exp()), a))
        .collect();
    // top-300 by score (partial sort is enough; downstream re-filters by conf).
    let k = NDET.min(na);
    scored.select_nth_unstable_by(k - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
    scored.truncate(k);
    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut d0 = vec![0f32; NDET * NATTR];
    for (row, &(score, a)) in scored.iter().enumerate() {
        let (l, t, r, b) = (boxes[a], boxes[na + a], boxes[2 * na + a], boxes[3 * na + a]);
        let s = ast[a];
        let base = row * NATTR;
        d0[base] = (ax[a] - l) * s; // x1
        d0[base + 1] = (ay[a] - t) * s; // y1
        d0[base + 2] = (ax[a] + r) * s; // x2
        d0[base + 3] = (ay[a] + b) * s; // y2
        d0[base + 4] = score;
        d0[base + 5] = 0.0; // class idx (nc=1)
        for c in 0..NM {
            d0[base + 6 + c] = coeff[c * na + a];
        }
    }
    (d0, d1, [NDET, NATTR, NM, ph, pw])
}
