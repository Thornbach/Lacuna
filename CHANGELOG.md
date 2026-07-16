# Changelog

All notable changes to Lacuna (FoliarToolbox) are documented in this file.

## [0.2.0] — 2026-07-06 to 2026-07-16

### Pipeline — Anomaly Detection & Clustering

**Fixed**
- **Reconstruction model silently failing on every leaf.** Root cause was a stale
  checkpoint trained on an older, wider U-Net architecture than the one currently
  in code — the mismatch crashed a background worker thread instantly and silently
  (release builds have no console, so the crash was invisible; it looked like an
  indefinite hang). Fixed by replacing the stale checkpoint with one matching the
  current architecture.
- **Tile-boundary seam artifacts.** Anomaly detection previously ran its
  confidence-threshold decision independently per tile; an anomaly spanning two
  tiles could pass in one tile and fail in the other, cutting it at the seam.
  Detection now stitches every tile's raw signal into one full-leaf map first and
  makes a single decision globally, so anomalies are judged as whole regions
  regardless of tile boundaries.
- **PatchCore (open-set detector) was silently skipped** whenever a few-shot head
  was configured, even though both are meant to run together (few-shot for known
  defect types, PatchCore specifically to catch novel/unseen ones). Both detectors
  now share one DINO feature pass and run side by side.
- **False positives on natural leaf margins** from an early version of hole
  detection — low model confidence at any jagged natural edge was briefly
  misread as a hole. Fixed by requiring genuine topological enclosure (not
  reachable from the image border) before flagging a hole, so natural margins are
  never affected.

**Added**
- **Hole detection**: the reconstruction model's predicted silhouette is now
  compared against the visible leaf to catch missing/background-colored tissue
  that the color/texture classifier alone couldn't see — both fully transparent
  "punched-through" holes and opaque background-colored holes are covered as two
  independently-detected cases.
- **"Novel (PatchCore)" cluster**: when PatchCore detects something the few-shot
  head misses, it's now surfaced as its own reviewable cluster instead of being
  silently dropped or double-counted — gated by a confidence multiplier so it
  doesn't just surface PatchCore's own background noise.
- **"Also run PatchCore" toggle** (off by default) — lets you opt into the
  dual-detector behavior only when a bank/meta pair is configured, rather than it
  running unconditionally.
- **Hard-negative stamping tool** ("Mark patch"): a Tile-Picker-style click-to-
  stamp workflow with a magnifier loupe and undo, for quickly capturing "this
  texture is not an anomaly" examples directly off the leaf canvas for the
  retrain flywheel.
- **Multi-select + bulk reassignment**: rubber-band drag-select on the canvas and
  Ctrl+click in the gallery both feed one selection set; selected regions get a
  highlighted bounding box on the canvas, and a right-click context menu offers
  Remove / Move to cluster / Clear — working for both single- and multi-region
  selections.
- **Undo for region removal** — a capped undo stack restores accidentally removed
  regions/clusters.
- Click anywhere on the canvas to clear the current selection.
- Reconstruction-preview overlay now correctly yields to a cluster's color
  wherever the two overlap, instead of visually blending.

**Changed**
- Widened cluster-name text fields across the stats table, reassign toolbar, and
  context menu — previously too narrow to display longer names like "Hole
  (reconstruction)".
- Synthetic erosion shapes for `lobe` and `focal_sector` damage modes now follow
  an organic, noise-modulated boundary instead of a perfect circle / mathematically
  straight cut, making training damage look more like real herbivory.
- **Segmentation cutting holes into the leaf interior**, not just the edges, on
  dark backgrounds — a background/near-background color check was zeroing out
  matching pixels unconditionally instead of only where they're actually
  reachable from the image border. Fixed with the same border-connectivity
  flood-fill approach already used for reconstruction hole detection: a pixel
  only gets treated as background if it's topologically connected to the
  border, not just color-similar.
- **Reconstruction quality regression**: the Pipeline was feeding leaves to the
  reconstruction model at 512px while the deployed checkpoint was trained at
  256px — silently degraded every prediction without erroring. Also ported over
  two techniques already used by the standalone Recon Infer tab but missing from
  the Pipeline: a small pre-damage nudge (keeps real herbivory inside the
  model's training distribution) and 4-rotation test-time augmentation.
- **Detected "holes" rendered as an opaque color fill**, which visually erased
  the exact thing that made them recognizable as holes (a genuinely transparent
  gap) — a correctly-detected hole would look like ordinary colored texture
  instead of an obvious gap once painted over. Hole regions now render as an
  outline only, leaving the underlying transparency visible.
- **Silent crashes looked like an indefinite hang.** A background pipeline
  worker that panics (e.g. from a model/checkpoint mismatch, or a GPU/driver
  issue on an unfamiliar machine) was previously indistinguishable from a slow
  computation — nothing ever surfaced, because release builds have no console
  and the panic message went nowhere. The worker now catches panics and
  surfaces them as a normal, readable error instead of a silent multi-hour wait.
- A cluster-id allocation bug: reassigning a region to a newly-typed cluster
  name only checked for collision against one of two reserved internal IDs, so
  a typed name could — in principle — silently collide with the "Novel
  (PatchCore)" sentinel. Now checks both.

**Added**
- **Hard-negative mining, at scale**: the shipped few-shot detection head was
  retrained against the FULL healthy-tile pool (46k+ tiles, up from a 4k-tile
  cap previously) to further suppress false positives on healthy tissue
  (veins, margins, natural texture) — healthy-tile false-firing dropped to 0%
  in evaluation, at a modest recall cost on two rarer defect families that's
  already compensated for via existing per-family thresholds. This retrained
  head is what ships as the default in this release.
- **PatchCore coreset bank bundled for the first time** — the open-set "catches
  anomaly types the head was never trained on" safety net (opt-in via the
  Pipeline tab's "Also run PatchCore" toggle) now ships with the app instead of
  requiring a separate manual setup.

### Pipeline — Tool Rail, Curation & Review Workflow

A full redesign of how you interact with detected regions, driven directly by
QA feedback that the old workflow "doesn't make sense" — Photoshop/GIMP-style
tools instead of a single "Mark patch" checkbox, and a curation flow that
can't silently lose your work.

**Added**
- **Left-side tool rail**, integrated into the same panel as the source/output
  folder pickers (not a separate floating panel): Select, Mark Healthy, Brush,
  Lasso select, and Eyedropper (inspect), each with its own icon and settings
  shown directly beneath it.
- **Brush tool**: paint a freeform region using an existing cluster's color
  (clickable color-swatch picker, or type a new family name), Square or Circle
  shape, adjustable size (also ctrl+scroll while painting). A stroke that
  touches an existing region of the same cluster extends it; a stroke that
  touches nothing creates a new region. Replaces the old fixed-size "stamp a
  square" tool for marking anomalies.
- **Standing rule: touching regions of the same cluster are always treated as
  one region** — previously a visually-continuous blob could be reported as
  several fragmented regions with split area stats (and an Eyedropper reading
  that flipped identity depending on which pixel you hovered). Now enforced
  everywhere, not just for freshly-painted brush strokes.
- **Lasso select**: drag a freeform outline to multi-select every region whose
  center falls inside it — feeds the same Confirm/Reject/Reassign actions as
  box-select.
- **Eyedropper**: hover a region to see its cluster, area, and review status,
  without selecting or modifying anything.
- **Universal zoom/pan**: scroll to zoom (any tool), hold middle-mouse to pan
  (any tool) — no dedicated tool needed, and zooming out below 100% now
  actually works (previously clamped at fit-to-window).
- **Immediate, per-action saving**: every Confirm, Reject, and Reassign now
  writes to disk the instant it happens, the same way the stamp tools already
  did — closes the exact gap that let curation work get silently lost if the
  app closed (or the pipeline re-ran) before a separate "Save" button was
  clicked. A live "N confirmed · M rejected · K unreviewed" counter replaces
  the old total absence of a running total, and the old Save button is now a
  "Confirm all remaining" bulk accelerator rather than the only path to disk.
- **Explicit Confirm action** (button, right-click menu, and Enter key) for
  "I reviewed this and it's correct" — previously only Reject had an explicit
  gesture; confirmation was only ever implicit.
- **Cluster filter dropdown** in the review tab — pick a cluster to review
  directly instead of needing to click an existing example of it first.
  Regions in the gallery are also grouped by cluster, with a header per group.
- **Right-side panel reorganized into tabs** (Metrics / Clusters / Curate /
  Log) instead of one long scrolling column mixing leaf stats, cluster stats,
  curation controls, retrain controls, and the run log together.
- **In-place retrain**: a "Retrain from curations" button lives directly in
  the Pipeline tab now — fine-tune the detection head from what you've just
  curated without switching tabs or re-picking folders. Writes a new sibling
  file (never overwrites the head you're using) and offers a one-click
  "Use this head now" once it finishes.
- **Clustering looseness controls** — DBSCAN's radius and minimum-points are
  now adjustable (Settings → Pipeline) instead of fixed; lower values produce
  more, smaller clusters. Only affects the PatchCore-only detection path.

**Changed**
- Cutout-edge and background-chroma-rejection sliders now default to 0
  (loosest/most inclusive setting).

### App Shell

**Changed**
- **The left navigation rail has been replaced with a top menu bar**
  (File / Train / Tools, plus a standalone Settings button) — a more
  traditional, less space-hungry layout that frees up horizontal room for
  every tab's own content. File groups the process/analyze tabs (Pipeline,
  Leaf Seg, Morphology, Recon Infer); Train groups the two training workflows
  (Recon Train, Train Detector); Tools groups the three standalone utilities
  (Tile Picker, Eroder, Sorter).

### Reconstruction Model Training

**Fixed**
- **Optimizer momentum bug**: the trainer used `beta_1=0.5` (a GAN-stabilization
  setting inherited by copy-paste) despite having no adversarial loss at all —
  this was throwing away Adam momentum that would otherwise help push through
  loss plateaus. Changed to the standard `beta_1=0.9`. Confirmed by the user to
  produce a more stable, still-improving training curve after the change.

**Added**
- **Cosine learning-rate schedule**: LR now decays smoothly from the base rate
  down to a configurable floor (default 5%) by the final epoch, instead of
  staying flat for the entire run. A floor of `1.0` disables decay entirely.
- **Proper resume-from-checkpoint support**: training can now genuinely continue
  a run rather than restart — the epoch loop, LR schedule, and "best IoU so far"
  bookkeeping all pick up from the correct point instead of resetting.
  Checkpoints now save a small sidecar file recording their epoch/IoU, so future
  resumes auto-fill the resume settings instead of requiring the user to
  remember and re-enter them by hand.
- **Hole-consistency loss**: a new loss term penalizes the model for predicting
  small isolated "holes" inside its own predicted leaf silhouette — a pattern
  ordinary per-pixel losses barely react to (too few pixels to move aggregate
  loss/IoU) but which showed up visibly in sample-grid previews on the
  best-performing checkpoint, and which would also cause false-positive hole
  detections downstream in the Pipeline. Configurable weight, disabled by
  setting it to 0.

### Removed — Legacy GAN Trainer

The old adversarial ("GAN Train") reconstruction trainer and its tab have been
fully removed; it was superseded by the currently-used non-adversarial trainer
("Recon Train") some time ago and was no longer in use.

- Removed the GAN Train tab and its entire UI.
- Removed the GAN training loop, discriminator network, and all
  adversarial-loss-specific code.
- Removed the `--recon-bench` CLI benchmark (a head-to-head comparison against
  the now-removed GAN trainer).
- Kept everything actually shared with the rest of the app fully intact: the
  U-Net architecture, device/backend helpers, damage/erosion utilities, metrics,
  and the sample-grid visualization — all still used by the active "Recon Train"
  tab and by the Pipeline's hole detection.
- The `--recon-train` CLI entry point (the one that trains the currently
  deployed checkpoint) is unaffected.

### Notes

- All changes above compile cleanly (`cargo build --release`, zero errors).
  The earlier Pipeline detection/clustering work (hole detection, dual-detector,
  multi-select tooling, the July 6–10 fixes) has been exercised in real use.
  **The Pipeline tool-rail/curation redesign and the App Shell menu bar (the
  "Tool Rail, Curation & Review Workflow" and "App Shell" sections above) have
  NOT yet been exercised in the running app by a human** — they're new this
  release and build-verified only. If something in that area looks off,
  that's the first place to check.
- The retrained few-shot head and the newly-bundled PatchCore bank are both
  new artifacts as of this release — the head's improvement (healthy-tile
  false-firing 23%→0%) is from offline evaluation, not yet a head-to-head
  comparison inside the running app.
