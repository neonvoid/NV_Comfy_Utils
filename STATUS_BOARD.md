# Status Board — NV_Comfy_Utils

> Auto-managed by `/handoff`. Content is never deleted — old entries move to ARCHIVE.md.
> Last updated: 2026-04-03

## Resume Context
<!-- Rewritten each `/handoff` run. What does a cold-start agent need RIGHT NOW? -->

- **Current focus:** Full body character swap artifact debugging. Multiband Laplacian dark halo on dark backgrounds is the remaining issue — try CropColorFix `composite_mode=hard`.
- **Critical files:** `src/KNF_Utils/vace_debug_preview.py`, `src/KNF_Utils/mask_processing_config.py`, `src/KNF_Utils/crop_color_fix.py`, `node_notes/guides/VACE_INPAINT_NODE_AUDIT.md`
- **Known blockers:** None
- **Environment notes:** 3 files with uncommitted cleanup (debug print removal). Research outputs in `node_notes/research/2026-04-03_rt-mask-edit-tool-v1/`. Read VACE_INPAINT_NODE_AUDIT.md for comprehensive param/artifact reference.

## Pulse
<!-- Last 2 session summaries, newest first. Older entries roll to Workstream Details. -->

### 2026-04-02/03 — Runtime debugging + audit doc + artifact research [coding + research]
- **Done:**
  - Runtime tested all renamed params — found and fixed ComfyUI widget position-shift bug (old workflow values land in wrong slots when param order changes)
  - Removed all deprecated backward-compat params (37 across 6 files) — old workflows won't load, clean UI
  - Fixed VaceDebugPreview: crop blend mask coordinate projection (crop-space vs full-frame auto-detection), stitcher input added, PIL text labels on all panels, expansion_info with suggestions engine (GREY VISIBLE, STITCH OVERREACH, OVERMASKING RISK, CLIP RISK, DARK SEAM RISK)
  - Fixed stale `apply_mask_config` docstring, removed all debug prints, removed dead `_OLD_TO_NEW`/`resolve_deprecated` code
  - Created VACE_INPAINT_NODE_AUDIT.md (2,128 lines) — complete parameter audit of 4 nodes with visual intuition guide, four-region framework (R0⊆Rv⊆Rc⊆Rs), 6 artifact patterns with reproduction/fix recipes, shot type cheat sheet, CropColorFix interaction table with dark/bright BG presets
  - Multi-AI research: color correction deep dive (CropColorFix Step 2 is constant scalar not exponential), real-time mask editor brainstorm (PySide6 + cached op graph MVP), artifact catalog with diagnostic decision tree
  - All research saved to `node_notes/research/2026-04-03_rt-mask-edit-tool-v1/` (10 files, ~120KB)
- **Decisions:** Remove deprecated params entirely — no backward compat (D-006). `forceInput:True` doesn't preserve old widget values (D-007). CropColorFix `boundary_lock` doesn't darken — it exposes what multiband already darkened (D-008). Use `crop_expand_px` for spatial coverage, NOT CropColorFix `blend_expansion` (D-009).
- **Blockers:** None
- **Next:** V1 agent-assisted param suggestions (extend VaceDebugPreview with BG luminance + boundary contrast analysis). Real-time mask editor MVP (PySide6). Full body swap shot still has residual dark halo from multiband Laplacian bleed — try `composite_mode=hard` on CropColorFix.

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
| A  | Mask Pipeline UX Refactor | ACTIVE | 2026-04-03 | Renames done, deprecated removed, debug preview working. Audit doc (2,128 lines) complete. |
| B  | Edge-of-Frame Fix | ACTIVE | 2026-04-03 | Crop clamp + reflection/zeros padding — runtime tested, working |
| C  | Clothing/Bag Swap Pipeline | ACTIVE | 2026-04-03 | Full body swap on dark BG — multiband dark halo is remaining artifact |
| D  | Real-Time Mask Editor | STAGED | 2026-04-03 | Research complete — PySide6 + cached op graph MVP. Not started. |

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
- Isotropic mask expansion (`vace_input_grow_px`) expands in ALL directions — problematic when content only extends in one direction
- ComfyUI `widgets_values` maps by POSITION not by NAME — renaming params scrambles old workflow values
- CropColorFix `composite_mode=multiband` pulls low-freq dark BG energy into character edges via Laplacian pyramid — use `hard` or `blend_pixels≤4` on dark backgrounds
- `crop_expand_px` only affects stitch when `crop_stitch_source=processed` or `hybrid` — has NO effect in `tight` mode

## Project Decisions Index
<!-- Numbered decisions with lifecycle status. Never renumber IDs. -->

| ID | Date | Status | Workstream | Description |
|----|------|--------|-----------|-------------|
| D-001 | 2026-04-01 | ACTIVE | A | Prefixed naming convention: `cleanup_*`, `crop_*`, `vace_*` (not double-underscore) |
| D-002 | 2026-04-01 | ACTIVE | A | One config node with categorized params (not split into multiple nodes, not modes/presets) |
| D-003 | 2026-04-01 | SUPERSEDED → D-006 | A | `resolve_deprecated()` uses value equality — replaced by full removal of deprecated params |
| D-004 | 2026-04-01 | ACTIVE | B | Crop clamped to frame bounds (no canvas extension). Isotropic shrink preserves aspect ratio. |
| D-005 | 2026-04-01 | ACTIVE | B | Stitch inverse warp uses `zeros` padding (not reflection) — compositing is mask-gated so zeros are invisible |
| D-006 | 2026-04-02 | ACTIVE | A | Remove all deprecated params — no backward compat for old workflows. Clean UI. |
| D-007 | 2026-04-02 | ACTIVE | A | `forceInput:True` does NOT preserve old widget values — ComfyUI treats as connection-only inputs |
| D-008 | 2026-04-03 | ACTIVE | C | `boundary_lock` restores original pixels via `torch.where`, doesn't darken — exposes what multiband created |
| D-009 | 2026-04-03 | ACTIVE | C | Use `crop_expand_px` for spatial coverage, NOT CropColorFix `blend_expansion` (causes dark borders) |

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
- **2026-04-02/03 | coding + research**
  Outcome: Full body swap on dark BG — identified 6 artifact patterns. CropColorFix multiband Laplacian bleed is primary remaining issue. Four-region framework (R0⊆Rv⊆Rc⊆Rs) established.
  Decision: Use `crop_expand_px` for spatial coverage not `blend_expansion` (D-009). `boundary_lock` exposes not darkens (D-008). Multiband with `blend_pixels>8` on dark BGs = dark halo.
  Next: Try CropColorFix `composite_mode=hard`. V1 agent-assisted param suggestions.

### D. Real-Time Mask Editor
**Current state:** STAGED — research complete, not started
**Goal:** Standalone tool for real-time visual mask parameter feedback (PySide6 + cached op graph)
**Key files:** `node_notes/research/2026-04-03_rt-mask-edit-tool-v1/` (10 research files, README with architecture decisions)
**Active constraints:** Must reuse `mask_ops.py` for production parity

**Milestones:**
- 2026-04-03 — Research complete: architecture decided (PySide6 + cached op graph MVP, two-speed long-term)

**History:**
- **2026-04-03 | research**
  Outcome: Multi-AI brainstorm (Codex+Gemini, 2 rounds) produced 3 architecture options. PySide6 + cached op graph selected for MVP. Two-speed (GPU approximation + Python validation) for long-term.
  Decision: PySide6 over web app (reuses mask_ops.py directly), cached op graph (slider invalidates only downstream), constraint-driven sliders as stretch goal.
  Next: Build MVP — load video+mask, real-time param sliders, export JSON to ComfyUI.

## Global Timeline
<!-- Thin chronological index of project-wide events. NOT per-workstream progress. -->

- **2026-04-01:** STATUS_BOARD initialized. Mask pipeline UX refactor implemented (9 files, +719/-300 lines). Edge-of-frame fix implemented. NV_VaceDebugPreview node created.
- **2026-04-02/03:** Deprecated params removed (37 across 6 files). VACE_INPAINT_NODE_AUDIT.md created (2,128 lines). Artifact research complete (10 files, ~120KB). Real-time mask editor research complete.

## Archived Workstreams Index
<!-- Pointers to workstreams moved to ARCHIVE.md. -->

*No archived workstreams.*
