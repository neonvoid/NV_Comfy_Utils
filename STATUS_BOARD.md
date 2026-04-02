# Status Board — NV_Comfy_Utils

> Auto-managed by `/handoff`. Content is never deleted — old entries move to ARCHIVE.md.
> Last updated: 2026-04-01

## Resume Context
<!-- Rewritten each `/handoff` run. What does a cold-start agent need RIGHT NOW? -->

- **Current focus:** Mask pipeline UX refactor — renames shipped, needs runtime testing. Also edge-of-frame crop/warp fixes need testing.
- **Critical files:** `src/KNF_Utils/mask_processing_config.py`, `src/KNF_Utils/vace_control_video_prep.py`, `src/KNF_Utils/inpaint_crop.py`, `src/KNF_Utils/vace_debug_preview.py` (NEW), `src/KNF_Utils/inpaint_stitch.py`
- **Known blockers:** None — all syntax-verified, needs ComfyUI restart + runtime test
- **Environment notes:** 11 files changed (+719/-300 lines), uncommitted. New node NV_VaceDebugPreview registered but untested.

## Pulse
<!-- Last 2 session summaries, newest first. Older entries roll to Workstream Details. -->

### 2026-04-01 — Mask pipeline UX refactor + edge-of-frame fixes [refactor]
- **Done:**
  - Renamed all mask params across 9 files: `mask_erode_dilate`→`crop_expand_px`, `mask_fill_holes`→`cleanup_fill_holes`, etc. Categories: `cleanup_*`, `crop_*`, `vace_*`
  - Promoted `mask_grow`→`vace_input_grow_px` and `halo_pixels`→`vace_halo_px` into config bus
  - Disambiguated `stitch_source` → `crop_stitch_source` (InpaintCrop2) and `vace_stitch_source` (VaceControlVideoPrep)
  - Full backward compat via deprecated optional params + `resolve_deprecated()` helper
  - Created NV_VaceDebugPreview node (side_by_side/overlay/grid + expansion_info summary)
  - Fixed crop_for_inpaint: hard clamp to frame bounds + isotropic shrink (no more canvas extension)
  - Fixed CoTracker warp: `padding_mode='border'`→`'reflection'`
  - Fixed stitch inverse warp: `padding_mode='border'`→`'zeros'` (mask-gated compositing)
  - Added input validation (bbox_w/h, target_w/h > 0), `assert`→`raise ValueError`, bounding_box_mask spatial check
  - Created VACE_MASK_FAQ.md and VACE_INPAINT_NODE_MAP.md documentation
  - Added shot categories (A-E) and agent debugging protocol to FAQ
- **Decisions:** Prefixed naming convention (D-001). One config node with categories, not split (D-002). `resolve_deprecated()` uses value equality — known edge case with intentional-default-reset (D-003).
- **Blockers:** None — needs runtime test
- **Next:** Restart ComfyUI, test renamed params load correctly with old workflows, test NV_VaceDebugPreview renders, test edge-of-frame crop clamp, update CHANGELOG.md

## Active Workstreams

| ID | Name | Status | Last Touch | Key Context |
|----|------|--------|-----------|-------------|
| A  | Mask Pipeline UX Refactor | ACTIVE | 2026-04-01 | Param renames + config reorg + debug preview — implementation done, needs runtime test |
| B  | Edge-of-Frame Fix | ACTIVE | 2026-04-01 | Crop clamp + reflection/zeros padding — syntax verified, needs runtime test |
| C  | Clothing/Bag Swap Pipeline | ACTIVE | 2026-04-01 | Production bag/jacket swap workflows driving the above fixes |

### Status Tags
- **ACTIVE** — Currently being worked on
- **STAGED** — Ready to start, not yet begun
- **PAUSED** — Intentionally on hold (note why)
- **STABLE** — Shipped/working, only touch for bugs
- **WATCHING** — Monitoring external dependency
- **STALE** — No activity for 14+ days (auto-flagged)

## System Constraints
<!-- Cross-cutting platform rules that affect all workstreams. -->

- ComfyUI caches Python modules — code changes require server restart to take effect
- `padding_mode='border'` in `grid_sample` causes edge-stretching artifacts — use `reflection` or `zeros`
- VACE neutral fill must be 0.5 (not 0.6/#999999) — WanVaceToVideo centers by -0.5
- `feather_blocks` must NEVER be 0 — hard mask edges cause Gibbs ringing in VAE
- Isotropic mask expansion (`mask_grow`) expands in ALL directions — problematic when content only extends in one direction

## Project Decisions Index
<!-- Numbered decisions with lifecycle status. Never renumber IDs. -->

| ID | Date | Status | Workstream | Description |
|----|------|--------|-----------|-------------|
| D-001 | 2026-04-01 | ACTIVE | A | Prefixed naming convention: `cleanup_*`, `crop_*`, `vace_*` (not double-underscore) |
| D-002 | 2026-04-01 | ACTIVE | A | One config node with categorized params (not split into multiple nodes, not modes/presets) |
| D-003 | 2026-04-01 | PROVISIONAL | A | `resolve_deprecated()` uses value equality — edge case: can't distinguish intentional-default from never-touched. May switch to presence-based. |
| D-004 | 2026-04-01 | ACTIVE | B | Crop clamped to frame bounds (no canvas extension). Isotropic shrink preserves aspect ratio. |
| D-005 | 2026-04-01 | ACTIVE | B | Stitch inverse warp uses `zeros` padding (not reflection) — compositing is mask-gated so zeros are invisible |

### Decision Statuses
- **ACTIVE** — Currently in effect
- **PROVISIONAL** — Experimental, may be revised
- **SUPERSEDED → D-NNN** — Replaced by another decision

## Workstream Details
<!-- Per-workstream context: state, goals, key files, decisions, milestones, history. -->
<!-- Read ONLY the workstream you're about to touch. -->

### A. Mask Pipeline UX Refactor
**Current state:** ACTIVE — implementation complete, awaiting runtime test
**Goal:** Rename mask params by stage/intent, promote mask_grow+halo to config, add debug preview node
**Key files:** `mask_processing_config.py`, `vace_control_video_prep.py`, `inpaint_crop.py`, `vace_debug_preview.py`, `mask_pipeline_viz.py`, `latent_inpaint_crop.py`, `temporal_mask_stabilizer.py`, `mask_tracking_bbox.py`
**Active constraints:** Backward compat required — old workflows must load

**Milestones:**
- 2026-04-01 — All 9 files renamed + NV_VaceDebugPreview created, syntax verified

**History:**
- **2026-04-01 | refactor**
  Outcome: Renamed all mask params across 9 files with backward compat, promoted mask_grow+halo to config, created NV_VaceDebugPreview
  Decision: Prefixed naming (D-001), one config node (D-002), value-equality deprecation (D-003)
  Next: Runtime test with ComfyUI restart, test old workflows, update CHANGELOG

### B. Edge-of-Frame Fix
**Current state:** ACTIVE — implementation complete, awaiting runtime test
**Goal:** Eliminate pixel-stretching artifacts when crop/warp extends beyond frame boundary
**Key files:** `inpaint_crop.py` (crop_for_inpaint), `cotracker_bridge.py`, `inpaint_stitch.py`
**Active constraints:** Cannot break CoTracker expand-crop-trim logic

**Milestones:**
- 2026-04-01 — Three-fix approach: crop clamp + reflection forward warp + zeros inverse warp

**History:**
- **2026-04-01 | coding**
  Outcome: Fixed crop (hard clamp + isotropic shrink), CoTracker (reflection), stitch (zeros). Multi-AI reviewed — approved with concerns.
  Decision: Crop clamp over canvas extension (D-004), zeros over reflection for inverse warp (D-005)
  Next: Runtime test — verify no stretching artifacts remain

### C. Clothing/Bag Swap Pipeline
**Current state:** ACTIVE — production use driving bug discovery
**Goal:** VACE inpainting for bag replacement and jacket swap on fashion video
**Key files:** Workflow files in `Z:\DerekWorkspace\0327_ClothingSwap\bags\`
**Active constraints:** Shot-dependent mask settings — no one-size-fits-all preset

**Milestones:**
- 2026-03-31 — Bag swap workflow identified grey artifact + overmasking issues
- 2026-04-01 — Jacket swap identified shoulder bleed from isotropic expansion

**History:**
- **2026-03-31 to 2026-04-01 | research + coding**
  Outcome: Debugged grey artifacts (mask_grow too small), overmasking (expansion too large), shoulder bleed (isotropic expansion into garment boundary). Created VACE_MASK_FAQ with 5 shot categories.
  Decision: bbox mode for object insertion (bag), as_is with tight settings for garment replacement (jacket). Different shot types need fundamentally different param strategies.
  Next: Test with updated param renames, use NV_VaceDebugPreview to verify mask coverage

## Global Timeline
<!-- Thin chronological index of project-wide events. NOT per-workstream progress. -->

- **2026-04-01:** STATUS_BOARD initialized. Mask pipeline UX refactor implemented (9 files, +719/-300 lines). Edge-of-frame fix implemented. NV_VaceDebugPreview node created.

## Archived Workstreams Index
<!-- Pointers to workstreams moved to ARCHIVE.md. -->

*No archived workstreams.*
