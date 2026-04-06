# Status Board — NV_Comfy_Utils

> Auto-managed by `/handoff`. Content is never deleted — old entries move to ARCHIVE.md.
> Last updated: 2026-04-06

## Resume Context
<!-- Rewritten each `/handoff` run. What does a cold-start agent need RIGHT NOW? -->

- **Current focus:** Test latent tail path (`previous_chunk_tail_latent`) on a known-bad chunk seam. Compare against baseline and pixel-tail variants using NV_SeamAnalyzer metrics.
- **Critical files:** `src/KNF_Utils/vace_prepass_reference.py`, `src/KNF_Utils/seam_analyzer.py`, `src/KNF_Utils/inpaint_stitch.py`, `node_notes/guides/PIPELINE_KNOWN_TRUTHS.md`
- **Known blockers:** Silent stitch crash at full-res (3 fixes identified: VRAM cleanup, frame validation, NaN warp guard — not yet applied). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` not confirmed.
- **Environment notes:** Multi-AI known-truths deep-dive results in `~/.multi-ai/research/20260406-155057/` (4 files, ~74KB) — needs synthesis into PIPELINE_KNOWN_TRUTHS.md. KlingStitchAdapter deprecated in __init__.py.

## Pulse
<!-- Last 2 session summaries, newest first. Older entries roll to Workstream Details. -->

### 2026-04-05/06 — Chunk seam continuity: tail-frame VACE conditioning + latent tail bypass [coding + research]
- **Done:**
  - Built NV_SeamAnalyzer diagnostic node (PSNR, SSIM, optical flow, sharpness, color delta across chunk boundaries)
  - Added `previous_chunk_tail` (IMAGE) and `previous_chunk_tail_latent` (LATENT) inputs to NV_VacePrePassReference
  - Latent tail path bypasses VAE re-encode entirely — zero color drift. Validated via 3 multi-AI code reviews (edge cases: -0 slice, spatial validation, device alignment, batch guard, input fallback)
  - Created PIPELINE_KNOWN_TRUTHS.md — canonical reference for confirmed facts, dead ends, and constraints
  - 3 rounds of multi-AI debates (6+ model responses per round) on seam continuity architecture
  - Diagnosed silent stitch crash: VRAM fragmentation + NaN warp data from low CoTracker visibility
  - Confirmed identity_anchor (raw Kling) causes brightness shift when mixed with CropColorFix'd VACE output (D-010)
  - Confirmed committed noise (January 2026) failed for fine-detail drift; reference_latents is dead code for WAN (D-011)
  - Multi-AI deep-dive on known truths doc produced ~74KB of analysis (4 files, not yet synthesized)
- **Decisions:** Use sampler latent output as tail source, not pixel re-encode (D-012). Skip identity_anchor until color-normalized (D-010). Latent tail gets no frame_repeat and no ref_strength scaling (D-013). VACE late-block skip connections (10/40 blocks) are the mechanism for fine-detail control (D-014).
- **Blockers:** Silent stitch crash at 3840×2160 (VRAM cleanup + NaN warp guard fix identified but not yet applied). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` not yet confirmed as set.
- **Next:** Test latent tail path on known-bad seam. Apply stitch crash fixes. Synthesize known-truths deep-dive into final PIPELINE_KNOWN_TRUTHS.md. Run NV_SeamAnalyzer on all variants for objective comparison.

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


## Active Workstreams

| ID | Name | Status | Last Touch | Key Context |
|----|------|--------|-----------|-------------|
| A  | Mask Pipeline UX Refactor | ACTIVE | 2026-04-06 | Renames done, deprecated removed, debug preview working. Audit doc complete. |
| B  | Edge-of-Frame Fix | STABLE | 2026-04-03 | Crop clamp + reflection/zeros padding — runtime tested, working |
| C  | Clothing/Bag Swap Pipeline | ACTIVE | 2026-04-06 | Full body + head swap. CropColorFix `composite_mode=hard` working. Multi-pass color drift being addressed. |
| D  | Real-Time Mask Editor | STAGED | 2026-04-03 | Research complete — PySide6 + cached op graph MVP. Not started. |
| E  | Chunk Seam Continuity | ACTIVE | 2026-04-06 | Latent tail input built + code-reviewed. Awaiting first test. NV_SeamAnalyzer ready. |

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
- Latent tail must come from sampler output (post-trim, pre-decode) — pixel re-encode introduces compounding color drift
- Identity anchor requires color normalization before mixing with CropColorFix'd VACE refs — raw Kling causes brightness shift
- `reference_latents` conditioning key is dead code for WAN — only Flux/Lumina/Omnigen read it. Use VACE `vace_frames` mask=0
- Multi-pass pipelines compound ~7/255 color drift per pass — CropColorFix corrects per-pass but cannot prevent drift entering VACE conditioning
- Latent-space cropping is BROKEN for WAN 3D VAE — spatial+temporal CausalConv3d receptive field makes latent crop ≠ pixel crop

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
| D-010 | 2026-04-06 | ACTIVE | E | Skip identity_anchor until color-normalized — raw Kling brightness differs from CropColorFix'd VACE output |
| D-011 | 2026-04-06 | ACTIVE | E | Committed noise (Jan 2026) empirically failed for fine-detail drift; `reference_latents` dead code for WAN |
| D-012 | 2026-04-06 | PROVISIONAL | E | Use sampler latent output as tail source (bypasses VAE re-encode, zero color drift) — awaiting test |
| D-013 | 2026-04-06 | ACTIVE | E | Latent tail: no frame_repeat (already natural temporal), no ref_strength scaling (preserve exact state) |
| D-014 | 2026-04-06 | ACTIVE | E | VACE late-block skip connections (10/40 DiT blocks) confirmed as mechanism for fine-detail conditioning |

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
- **2026-04-01 | refactor (rolled from Pulse)**
  Outcome: Crop hard-clamp to frame bounds, CoTracker reflection padding, stitch zeros padding, input validation overhaul
  Decision: Crop clamped (D-004), stitch zeros (D-005)
  Next: Runtime test edge-of-frame artifacts

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

### E. Chunk Seam Continuity
**Current state:** ACTIVE — latent tail input built, awaiting first test
**Goal:** Eliminate fine-detail drift (hair, fabric, speculars) at chunk boundaries while avoiding color drift from VAE re-encoding
**Key files:** `vace_prepass_reference.py`, `seam_analyzer.py`, `node_notes/guides/PIPELINE_KNOWN_TRUTHS.md`
**Active constraints:** No identity_anchor until color-normalized. Latent tail = no repeat, no ref_strength scaling. Post-stitch tail prohibited (composite artifacts).

**Milestones:**
- 2026-04-05 — 3 rounds multi-AI debate converged on Approach A (mask=0 tail prepend via VACE late-block skip connections)
- 2026-04-06 — Pixel tail tested (brightness OK without anchor, color drift on multi-pass). Latent tail path implemented and code-reviewed.

**History:**
- **2026-04-05/06 | coding + research**
  Outcome: Built NV_SeamAnalyzer + latent tail input on VacePrePassReference. Pixel tail caused color drift from VAE re-encode; latent path bypasses entirely. 3 multi-AI debate rounds + 3 code reviews.
  Decision: Latent tail preferred over pixel tail (D-012). Skip identity_anchor (D-010). VACE late-block mechanism confirmed (D-014).
  Next: Test latent tail on known-bad seam. Apply stitch crash fixes. Synthesize known-truths deep-dive.

## Global Timeline
<!-- Thin chronological index of project-wide events. NOT per-workstream progress. -->

- **2026-04-01:** STATUS_BOARD initialized. Mask pipeline UX refactor implemented (9 files, +719/-300 lines). Edge-of-frame fix implemented. NV_VaceDebugPreview node created.
- **2026-04-02/03:** Deprecated params removed (37 across 6 files). VACE_INPAINT_NODE_AUDIT.md created (2,128 lines). Artifact research complete (10 files, ~120KB). Real-time mask editor research complete.
- **2026-04-05/06:** NV_SeamAnalyzer + latent tail input built. PIPELINE_KNOWN_TRUTHS.md created. 3 multi-AI debate rounds on chunk seam architecture. Committed noise (Jan 2026) and reference_latents confirmed as dead ends for WAN.

## Archived Workstreams Index
<!-- Pointers to workstreams moved to ARCHIVE.md. -->

*No archived workstreams.*
