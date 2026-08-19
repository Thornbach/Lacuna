# Third-party notices

Lacuna bundles and builds on the following third-party models and libraries. Their
licenses and required attributions are reproduced or referenced below.

---

## DINOv3 (Meta Platforms, Inc.)

**Built with DINOv3.**

Lacuna's anomaly-detection backbone is the DINOv3 ViT-B/16 model by Meta, reimplemented
in Burn and distributed as converted weights (`dino_weights.safetensors`).

DINOv3 is provided under the **DINOv3 License**, which grants a worldwide, royalty-free
right to use, reproduce, distribute, and create derivative works, subject to its terms.
When redistributing the weights or derivatives you must (1) include a copy of the DINOv3
License and (2) prominently display **"Built with DINOv3"** — both of which this
distribution does (see `LICENSES/DINOv3-LICENSE.md` and the app's About panel).

- License: https://ai.meta.com/resources/models-and-libraries/dinov3-license/
- Source: https://github.com/facebookresearch/dinov3

---

## Ultralytics YOLO (YOLO26-seg)

Leaf instance segmentation uses a model trained with **Ultralytics YOLO**, reimplemented
in Burn and distributed as converted weights (`yolo_weights.safetensors`).

Ultralytics YOLO is licensed under **AGPL-3.0**. Because Lacuna incorporates a
YOLO-derived model, distributing Lacuna carries AGPL-3.0 obligations: the corresponding
source of Lacuna must be made available to recipients under AGPL-3.0. (Ultralytics also
offers a separate commercial license for closed-source use.)

- License: https://www.gnu.org/licenses/agpl-3.0.html
- Source: https://github.com/ultralytics/ultralytics

---

## Reconstruction model

The intact-blade reconstruction U-Net (`recon/gen.mpk`) was trained by the Lacuna authors
and is distributed as part of Lacuna.

---

## Rust libraries

Lacuna is built on the Rust crate ecosystem. Key dependencies and their licenses:

| Crate | License |
|---|---|
| burn / burn-cuda / cubecl | MIT OR Apache-2.0 |
| egui / eframe / egui_plot / egui_extras | MIT OR Apache-2.0 |
| image | MIT OR Apache-2.0 |
| safetensors | Apache-2.0 |
| rayon, serde, walkdir, rand, dirs, rfd | MIT OR Apache-2.0 |
| ort (optional `ort-backend` feature only) | MIT OR Apache-2.0 (bundles ONNX Runtime, MIT) |

Full per-crate license texts can be generated with `cargo about` or `cargo-license`.

---

## Lacuna's own license

Given the Ultralytics YOLO (AGPL-3.0) dependency, Lacuna is released under
**AGPL-3.0-only**. The full text is in `LICENSE` at the repository root, and is
bundled in every distributed package.

Practically, for anyone handing this to colleagues or conference attendees: that
is distribution to third parties, so recipients are entitled to the corresponding
source of the version they received, under AGPL-3.0. Point them at the repository
(or ship `Lacuna-src-<commit>.zip` alongside the binary) rather than the binary
alone.
