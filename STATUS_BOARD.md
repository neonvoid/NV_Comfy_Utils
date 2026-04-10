# Status Board — NV_Comfy_Utils

> Auto-managed by `/handoff`. Content is never deleted — old entries move to ARCHIVE.md.
> Last updated: 2026-04-10c

## Resume Context
<!-- Rewritten each `/handoff` run. What does a cold-start agent need RIGHT NOW? -->

- **Current focus:** Build NV_VaceChunkedOrchestrator (workstream H). Single-queue-press chunked VACE inpainting. NV_FrequencyShapedNoise experiment shelved — flow-matching DiTs reject init-noise priors. TextureHarmonize is the production aesthetic-matching path.
- **Critical files:** `src/KNF_Utils/texture_harmonize.py` (production, MAD mode), `src/KNF_Utils/vace_latent_splice.py` (production), `src/KNF_Utils/frequency_shaped_noise.py` (SHELVED, kept for reference), future `src/KNF_Utils/vace_chunked_orchestrator.py`
- **Known blockers:** None.
- **Environment notes:** Multi uncommitted: texture_harmonize.py (MAD upgrade), frequency_shaped_noise.py (built+shelved), multi_model_sampler.py (gained noise input — keep), vace_prepass_reference.py (tail removed).

## Pulse
<!-- Last 2 session summaries, newest first. Older entries roll to Workstream Details. -->

### 2026-04-10c — Frequency-shaped noise experiment SHELVED + texture harmonize MAD upgrade [coding + research]
- **Done:**
  - Built NV_FrequencyShapedNoise node — full-frame pixel-space FFT → radial profile → shaped latent noise via deterministic linear filter. Wired into NV_MultiModelSampler via optional NOISE input (no SamplerCustomAdvanced swap needed).
  - Multi-AI bug review found CRITICAL fftshift mismatch (DC at center for profile extraction, DC at corners for shaping → entire spectrum was inverted). Plus DC bin domination (127x ratio was mostly image mean), linear amplitude blending instead of deterministic filter, no high-freq floor.
  - Fixed all 7 issues: fftshift alignment, DC removal (skip first 2 bins), power spectrum + sqrt for amplitude, log-domain blending with bounded clamp (gain range [0.5, 2.0]), deterministic linear filter F_out=H*F_white, per-channel renormalization, batched FFT.
  - Runtime tested at strength=1.0 with bug fixes: STILL severe rainbow checkerboard artifacts. The model is treating shaped noise as content to denoise, not as a stylistic prior.
  - Multi-AI final verdict: DEAD END. Both Codex and Gemini independently concluded the approach is fundamentally incompatible with WAN's flow-matching DiT architecture. No useful operating window between "ignored" (weak shaping) and "destabilizing" (catastrophic artifacts).
  - Texture harmonize gained MAD (Median Absolute Deviation) stat mode — replaces std as default for measuring texture spread. Robust to single bright outliers (eyelashes, specular highlights) that caused temporal strobing in tight crops as masks wiggled frame-to-frame.
- **Decisions:** NV_FrequencyShapedNoise SHELVED — fundamentally incompatible with flow-matching DiTs (D-029). Init-noise covariance changes are read as content by flow-matching models, not as priors. TextureHarmonize defaults to MAD instead of std — kills temporal strobing from outlier-driven ratio swings (D-030).
- **Blockers:** None.
- **Next:** Build NV_VaceChunkedOrchestrator. The aesthetic problem is now fully owned by post-processing (TextureHarmonize). Move on from prior-hacking experiments.

### 2026-04-10b — Texture harmonize + aesthetic conditioning research + prepass cleanup [coding + research]
- **Done:**
  - Built NV_TextureHarmonize node — Laplacian pyramid variance matching (sharpness/micro-contrast) + per-channel grain synthesis. Multi-AI reviewed (Codex+Gemini): fixed reversed pyramid indexing, tightened ratio clamp (0.25-3.0), batch-averaged ratios for temporal stability, per-frame grain seeding, soft mask weights.
  - Runtime tested: sharpness ratios 0.54/0.57 confirmed AI crop was ~2x too sharp. Grain stage correctly added nothing (clean source).
  - Cleaned up VacePrePassReference: removed tail inputs (previous_chunk_tail, previous_chunk_tail_latent, num_tail_frames, num_tail_latent_frames) — ~80 lines removed. Tail continuity now fully owned by VaceControlVideoPrep + NV_VaceLatentSplice.
  - Multi-AI deep dive: texture harmonization algorithms (Laplacian pyramids, FFT grain synthesis, differentiable JPEG). Validated approach matches Adobe Multi-scale Harmonization (2010) lineage.
  - Multi-AI deep dive: papers research (20+ papers reviewed). Key finds: Real-ESRGAN degradation model, Film Grain Rendering (Zhang 2023), DISTS metric, signal-dependent noise.
  - Multi-AI brainstorm: WAN DiT aesthetic conditioning injection points. Secondary VACE branch (sigma-gated) = best v1. Feature stats transfer (blocks 30-39) = best v2. Frequency-shaped noise = easy zero-risk experiment.
- **Decisions:** VacePrePassReference tail inputs removed — redundant with control video + splice path (D-026). Texture harmonize is post-decode only — VAE bottleneck prevents DiT-level grain/sharpness control (D-027). Frequency-shaped noise worth testing as complementary approach (D-028 PROVISIONAL).
- **Blockers:** None.
- **Next:** Build NV_FrequencyShapedNoise — pixel-space FFT of full frame → shaped latent noise for SamplerCustomAdvanced. Test if generation quality character shifts toward source aesthetic.


## Active Workstreams

| ID | Name | Status | Last Touch | Key Context |
|----|------|--------|-----------|-------------|
| A  | Mask Pipeline UX Refactor | ACTIVE | 2026-04-06 | Renames done, deprecated removed, debug preview working. Audit doc complete. |
| B  | Edge-of-Frame Fix | STABLE | 2026-04-03 | Crop clamp + reflection/zeros padding — runtime tested, working |
| C  | Clothing/Bag Swap Pipeline | ACTIVE | 2026-04-08 | Full body + head swap. CropColorFix validation added (D-020). Multi-pass workflow stabilizing. |
| D  | Real-Time Mask Editor | STAGED | 2026-04-03 | Research complete — PySide6 + cached op graph MVP. Not started. |
| E  | Chunk Seam Continuity | ACTIVE | 2026-04-10 | NV_VaceLatentSplice built + runtime validated. Zero-drift tail overlap confirmed. |
| F  | Kling API Chunking | ACTIVE | 2026-04-09 | type="first_frame" hints on tail refs. Debug logging added. Awaiting runtime test of API acceptance. |
| G  | Masking & VFI Pipeline Research | ACTIVE | 2026-04-09 | Mocha for sub-object, SAM3 for full body, MatAnyone for edge refinement. GIMM-VFI stays. NV_MatchInterpFrames + RetimePrep JSON persistence built. |
| H  | VACE Chunked Orchestrator | STAGED | 2026-04-10 | Architecture designed (4 multi-AI rounds). Single-queue-press chunked VACE inpainting with latent splice. |
| I  | Texture Harmonize + Aesthetic Conditioning | ACTIVE | 2026-04-10 | NV_TextureHarmonize w/ MAD stat mode (kills temporal strobing). Frequency-shaped noise SHELVED — incompatible with WAN flow-matching DiT. |

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
- Sampler-output latents and VAE-encoder-output latents are NOT interchangeable — different distributions cause color/sharpness shift when mixed in VACE conditioning
- `control_masks` (from VaceControlVideoPrep) goes ONLY to VacePrePassReference/WanVaceToVideo — never to CropColorFix or InpaintStitch
- Kling edit mode does NOT preserve original framing/camera — output cannot be stitched back onto original frames. Use as standalone output, not as InpaintStitch input.
- NV_VaceLatentSplice offset must account for VacePrePassReference ref frames: splice at [:16, ref_T:ref_T+tail_T], not [:16, :tail_T]
- Latent save MUST use native dtype — fp16 quantization causes visible noise bump artifacts when spliced into VACE conditioning
- VAE bottleneck destroys film grain, sensor noise, and compression artifacts — DiT cannot generate these. Texture harmonization must be post-decode.
- Laplacian pyramid level 0 = finest detail (highest frequency), last level = base residual (lowest frequency). Skip base levels from END, not from 0.
- WAN/VACE flow-matching DiTs are MORE brittle to init-noise structure than older U-Net DDPM models. Init noise must be near-isotropic Gaussian — covariance perturbations get read as content and produce structured failure modes (rainbow checkerboards). Aesthetic control must be post-decode, not init-shaping.
- TextureHarmonize: use MAD (default) instead of std for texture spread on tight crops where mask edges contain outlier pixels. std swings 20-40% from a single bright pixel and causes temporal strobing.

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
| D-012 | 2026-04-06 | SUPERSEDED → D-016 | E | Use sampler latent output as tail source — superseded by tail-as-control-video approach |
| D-013 | 2026-04-06 | ACTIVE | E | Latent tail: no frame_repeat (already natural temporal), no ref_strength scaling (preserve exact state) |
| D-014 | 2026-04-06 | ACTIVE | E | VACE late-block skip connections (10/40 DiT blocks) confirmed as mechanism for fine-detail conditioning |
| D-015 | 2026-04-07 | ACTIVE | E | Sampler-output latents ≠ VAE-encoder latents — different distributions cause artifacts when mixed |
| D-016 | 2026-04-07 | PROVISIONAL | E | Tail-as-control-video (mask=0 in control region) preferred — native VACE inpainting VCU pattern |
| D-017 | 2026-04-07 | ACTIVE | C,E | `control_masks` → VACE only. CropColorFix uses `cropped_mask_processed`. InpaintStitch uses `stitch_mask`. |
| D-018 | 2026-04-07 | ACTIVE | F | KlingStitchAdapter deprecated — Kling edit mode doesn't reliably preserve framing/camera for stitch-back |
| D-019 | 2026-04-07 | ACTIVE | F | No chunk overlap/crossfade — same framing instability that killed stitch makes crossfade produce ghosting |
| D-020 | 2026-04-08 | ACTIVE | C,E | CropColorFix must hard-fail on original≠generated frame count (was silently truncating, misaligning all frames) |
| D-021 | 2026-04-08 | ACTIVE | E | All VaceControlVideoPrep exit paths must apply tail prepend via _apply_tail() helper |
| D-022 | 2026-04-09 | PROVISIONAL | F | Tag last tail ref as type="first_frame" for chunk-start anchoring — API acceptance unverified |
| D-023 | 2026-04-10 | ACTIVE | E,H | Latent splice into vace_frames[:, :16] inactive channels — encoder-domain KSampler output directly replaces roundtrip-drifted tail |
| D-024 | 2026-04-10 | ACTIVE | E,H | Save latents as native dtype (fp32/bf16), never fp16 — quantization causes visible noise in splice path |
| D-025 | 2026-04-10 | PROVISIONAL | H | Orchestrator architecture: crop-space, CropColorFix inside loop, Kling refs chunk 0 only, last-chunk rollback, seed+chunk_idx |
| D-026 | 2026-04-10 | ACTIVE | E | VacePrePassReference tail inputs removed — redundant with VaceControlVideoPrep + VaceLatentSplice |
| D-027 | 2026-04-10 | ACTIVE | I | Texture harmonize must be post-decode only — VAE bottleneck prevents DiT-level grain/sharpness control |
| D-028 | 2026-04-10 | SUPERSEDED → D-029 | I | Frequency-shaped noise (pixel-space FFT of full frame → shaped latent noise) as complementary aesthetic conditioning |
| D-029 | 2026-04-10 | ACTIVE | I | NV_FrequencyShapedNoise SHELVED — flow-matching DiTs read init-noise covariance changes as content, not as priors. No usable operating window between "ignored" and "destabilizing". Post-processing (TextureHarmonize) is the right architectural answer. |
| D-030 | 2026-04-10 | ACTIVE | I | TextureHarmonize defaults to MAD (Median Absolute Deviation) instead of std for texture spread — robust to single-pixel outliers (eyelashes/highlights) that caused temporal strobing in tight crops as masks wiggled |

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
**Active constraints:** Deprecated params fully removed (D-006) — no backward compat. Old workflows must be re-wired.

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
**Current state:** STABLE — shipped and runtime tested
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
- **2026-04-02/03 | coding + research (rolled from Pulse)**
  Outcome: Deprecated params removed (37 across 6 files). VACE_INPAINT_NODE_AUDIT.md (2,128 lines). VaceDebugPreview fixed. Multi-AI artifact research (10 files, ~120KB).
  Decision: Remove deprecated params, no backward compat (D-006). forceInput doesn't preserve values (D-007).
  Next: composite_mode=hard for dark BG halo. Agent-assisted param suggestions.

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
**Current state:** ACTIVE — tail-as-control-video implemented, awaiting clean test
**Goal:** Eliminate fine-detail drift at chunk boundaries using VACE's native inpainting continuation (mask=0 preserved tail → mask=1 generation)
**Key files:** `vace_control_video_prep.py`, `vace_prepass_reference.py`, `seam_analyzer.py`, `node_notes/guides/PIPELINE_KNOWN_TRUTHS.md`
**Active constraints:** No identity_anchor until color-normalized. `control_masks`→VACE only. Tail overlap must be multiple of 4 (WAN 4k+1). Post-stitch tail prohibited.

**Milestones:**
- 2026-04-05 — 3 rounds multi-AI debate converged on Approach A (mask=0 tail prepend via VACE late-block skip connections)
- 2026-04-06 — Pixel tail tested (color drift), latent tail tested (domain mismatch). Both add artifacts.
- 2026-04-07 — Pivoted to tail-as-control-video (Approach B). VaceControlVideoPrep updated. Mask wiring clarified.

**History:**
- **2026-04-05/06 | coding + research**
  Outcome: Built NV_SeamAnalyzer + latent tail input on VacePrePassReference. Pixel tail caused color drift from VAE re-encode; latent path bypasses entirely. 3 multi-AI debate rounds + 3 code reviews.
  Decision: Latent tail preferred over pixel tail (D-012). Skip identity_anchor (D-010). VACE late-block mechanism confirmed (D-014).
  Next: Test latent tail on known-bad seam. Apply stitch crash fixes. Synthesize known-truths deep-dive.
- **2026-04-07 | coding + testing**
  Outcome: Latent tail tested — domain mismatch confirmed (D-015). Pivoted to tail-as-control-video in VaceControlVideoPrep (D-016). Critical review fixes applied. Mask wiring clarified (D-017).
  Decision: Sampler≠encoder latents (D-015). Tail-as-control-video preferred (D-016). control_masks→VACE only (D-017).
  Next: Clean end-to-end test of tail-as-control-video. Apply stitch fixes. Synthesize known-truths doc.
- **2026-04-08 | coding**
  Outcome: Stitch fixes applied (validation+VRAM+try/except). CropColorFix hard-fail on mismatch (D-020). VaceControlVideoPrep _apply_tail() on all exits (D-021). Audit doc +153 lines.
  Decision: Hard-fail > silent truncate (D-020). Consistent tail prepend (D-021).
  Next: Clean e2e test of tail-as-control-video. Runtime test Kling chunking.
- **2026-04-10 | coding + research**
  Outcome: Built NV_VaceLatentSplice — splices encoder-domain latents into VACE conditioning. Runtime tested on multi-chunk knight video, color consistency confirmed. Multi-AI reviewed. Fixed latent save (fp32) + load (weights_only=True).
  Decision: Latent splice into inactive channels (D-023). Native dtype save (D-024). Orchestrator architecture designed (D-025).
  Next: Build NV_VaceChunkedOrchestrator (Phase 1: frame slicing skeleton).

### F. Kling API Chunking
**Current state:** ACTIVE — type hints added + multi-AI reviewed, awaiting runtime test of API acceptance
**Goal:** Enable Kling API processing for videos >10s via sequential chunking with tail-frame reference continuity
**Key files:** `kling_edit_fork.py`, `kling_stitch_adapter.py` (deprecated)
**Active constraints:** No overlap/crossfade (D-019). Max 4 ref images total (API limit). Kling output is standalone — not stitchable (D-018).

**Milestones:**
- 2026-04-07 — Sequential chunk mode implemented. KlingStitchAdapter deprecated. Multi-AI reviewed.
- 2026-04-09 — OmniParamImage.type hints implemented. Multi-AI reviewed (Codex+Gemini approved, API compat flagged).

**History:**
- **2026-04-07 | coding**
  Outcome: Built chunk_mode on UploadPreview (auto-truncate + tail refs + next_chunk_start). Deprecated KlingStitchAdapter. Codex caught max_chunk_frames overshoot bug — fixed.
  Decision: No stitch-back (D-018), no crossfade (D-019). Hard cuts between chunks with ref-image consistency hints.
  Next: Runtime test on >10s video. Evaluate tail-frame ref effectiveness.
- **2026-04-07b | coding**
  Outcome: Built chunk_mode on UploadPreview (auto-truncate + tail refs + next_chunk_start). Deprecated KlingStitchAdapter. Codex caught max_chunk_frames bug.
  Decision: No stitch-back (D-018), no crossfade (D-019).
  Next: Runtime test >10s video. Evaluate tail-frame ref effectiveness.
- **2026-04-09 | coding**
  Outcome: Implemented OmniParamImage.type="first_frame" hints on tail ref images. Multi-AI reviewed (Codex+Gemini approved, API compat flagged). Debug logging added.
  Decision: Tag last tail ref as type="first_frame" for chunk-start anchoring (D-022, PROVISIONAL).
  Next: Runtime test chunked Kling with type= hints.

### G. Masking & VFI Pipeline Research
**Current state:** ACTIVE
**Goal:** Establish optimal mask generation + frame interpolation pipeline for WAN/VACE workflows
**Key files:** `src/KNF_Utils/match_interp_frames.py`, `src/KNF_Utils/temporal_retime.py`, `D:/DereksFiles/.../mochapro/08_genai_mask_pipeline.md`
**Active constraints:** MatAnyone v2 = whole-person only (not sub-object). SAM3 chunking exists but untested on long videos.

**Milestones:**
- 2026-04-09 — Mocha-for-GenAI guide written. NV_MatchInterpFrames built. RetimePrep/Restore gains JSON persistence.

**History:**
- **2026-04-09 | research + coding**
  Outcome: Mocha Pro guide, segmentation/matting deep dive, VFI SOTA review. NV_MatchInterpFrames + RetimePrep JSON persistence built. SAM3 chunking confirmed built-in.
  Decision: Mocha for sub-object, SAM3 for full body, MatAnyone whole-person only. GIMM-VFI stays.
  Next: Runtime test SAM3→MatAnyone pipeline. Test chunked tracking on 600+ frames.

### H. VACE Chunked Orchestrator
**Current state:** STAGED — architecture designed, implementation not started
**Goal:** Single ComfyUI node that processes entire video through chunked VACE inpainting in one queue press
**Key files:** `vace_latent_splice.py` (built), `latent_guidance.py` (save/load fixed), future `vace_chunked_orchestrator.py`
**Active constraints:** Crop-space only (InpaintCrop/Stitch external). CropColorFix must run inside loop. Kling refs chunk 0 only. Last-chunk rollback (retract start frame, not pad).

**Architecture (from 4 multi-AI rounds):**
- Hybrid orchestrator: single node, internal Python loop, imports shared helpers
- State across chunks: prev_tail_pixels (corrected) + prev_tail_latents (encoder-domain)
- Overlap 12 frames (3 latent frames of velocity context)
- Seed = base_seed + chunk_idx
- Last-chunk rollback: retract start frame so WAN always sees valid 4k+1

**Milestones:**
- 2026-04-10 — Architecture designed. NV_VaceLatentSplice built + runtime validated.

**History:**
*First session — architecture design only.*

### I. Texture Harmonize + Aesthetic Conditioning
**Current state:** ACTIVE — NV_TextureHarmonize is the production texture-matching path. NV_FrequencyShapedNoise SHELVED.
**Goal:** Make AI-generated crops visually match source footage's texture quality (sharpness, grain, micro-contrast) via post-processing
**Key files:** `texture_harmonize.py` (production), `frequency_shaped_noise.py` (shelved, kept for reference), `multi_model_sampler.py` (has noise input — keep for future use)
**Active constraints:** VAE bottleneck prevents DiT-level grain/sharpness — post-decode is the ONLY viable control surface for texture characteristics. Init-noise priors do NOT work on flow-matching DiTs (D-029).

**Architecture (production):**
- Laplacian pyramid variance matching — sharpness + micro-contrast per frequency band
- Per-channel grain synthesis — additive only, frame-seeded, MAD-based stat for outlier robustness
- Pipeline position: VAE Decode → CropColorFix → NV_TextureHarmonize → InpaintStitch

**Shelved experiment — NV_FrequencyShapedNoise:**
- Approach: pixel-space FFT of full source frame → radial power spectrum → shaped initial noise via deterministic linear filter
- Why it failed: WAN flow-matching DiT reads init-noise covariance changes as content, not as priors
- Empirical: NO useful operating window. Weak shaping = ignored. Strong shaping = rainbow checkerboard catastrophic failure.
- Bug fixes that didn't save it: fftshift alignment, DC removal, log-domain blending, bounded gain [0.5, 2.0], deterministic linear filter F=H*F_white, per-channel renormalization. ALL applied. Still failed.
- Theoretical reason: DDPM U-Nets have SDE drift that washes out off-distribution starts. Flow-matching ODEs don't — bad start = bad trajectory all the way through.
- The node + the optional noise input on NV_MultiModelSampler are kept for potential future use with non-flow-matching architectures, but should NOT be wired in current production workflows.

**Lessons learned (preserve for future):**
- Modern DiTs are MORE brittle to init-noise structure than older U-Nets, not less
- The "control noise to control output" intuition is wrong for flow matching
- Aesthetic control on flow matching = post-decode pixel ops, not prior hacking
- If aesthetic conditioning is ever needed inside the model, look at FreeU-style feature manipulation (block 30-39 hidden state shaping), not noise shaping
- Secondary sigma-gated VACE branch is the most promising untested approach for in-model aesthetic conditioning

**Research findings (20+ papers reviewed):**
- Our texture harmonize approach matches Adobe Multi-scale Harmonization (Sunkavalli 2010) lineage
- Real-ESRGAN degradation model useful as reverse-application reference
- DISTS/A-DISTS metrics best for texture similarity evaluation
- Signal-dependent noise (heteroscedastic: variance scales with luminance) = key upgrade for grain realism

**Milestones:**
- 2026-04-10 — NV_TextureHarmonize built + runtime tested (sharpness ratios 0.54/0.57). Multi-AI reviewed + 5 fixes applied.
- 2026-04-10 — NV_FrequencyShapedNoise built, fully debugged, empirically tested, shelved. Knowledge captured.
- 2026-04-10 — TextureHarmonize gained MAD stat mode for outlier robustness.

**History:**
- **2026-04-10 | coding + research (shelved experiment)**
  Outcome: NV_FrequencyShapedNoise built and tested through 2 multi-AI review rounds + bug fixes. Empirically confirmed dead end for flow-matching DiTs. TextureHarmonize gained MAD mode.
  Decision: SHELVE init-noise shaping for WAN/VACE (D-029). MAD as default texture statistic (D-030). Aesthetic control via post-decode only.
  Next: Build NV_VaceChunkedOrchestrator. Aesthetic problem is solved by TextureHarmonize.

## Global Timeline
<!-- Thin chronological index of project-wide events. NOT per-workstream progress. -->

- **2026-04-01:** STATUS_BOARD initialized. Mask pipeline UX refactor implemented (9 files, +719/-300 lines). Edge-of-frame fix implemented. NV_VaceDebugPreview node created.
- **2026-04-02/03:** Deprecated params removed (37 across 6 files). VACE_INPAINT_NODE_AUDIT.md created (2,128 lines). Artifact research complete (10 files, ~120KB). Real-time mask editor research complete.
- **2026-04-05/06:** NV_SeamAnalyzer + latent tail input built. PIPELINE_KNOWN_TRUTHS.md created. 3 multi-AI debate rounds on chunk seam architecture. Committed noise (Jan 2026) and reference_latents confirmed as dead ends for WAN.
- **2026-04-07:** Latent tail domain mismatch confirmed (sampler≠encoder). Pivoted to tail-as-control-video (VaceControlVideoPrep). Mask wiring rules formalized (D-017).
- **2026-04-07:** NV_KlingUploadPreview gains sequential chunk mode (auto-truncate + tail-frame refs). KlingStitchAdapter deprecated.
- **2026-04-08:** Stitch crash fixes applied (validation+VRAM+debug). CropColorFix hard-fail on mismatch. VaceControlVideoPrep tail consistency fix. VACE_INPAINT_NODE_AUDIT.md addendum (+153 lines, 5 sections).
- **2026-04-09:** Masking research arc: Mocha Pro guide (08_genai_mask_pipeline.md), segmentation/matting deep dive, VFI SOTA review. NV_MatchInterpFrames node built. RetimePrep/Restore gains JSON persistence + config_only mode.
- **2026-04-10:** NV_VaceLatentSplice built + runtime validated (zero-drift tail overlap). Latent save/load security + precision fixes. Orchestrator architecture designed (4 multi-AI rounds). New workstream H: VACE Chunked Orchestrator.
- **2026-04-10:** NV_TextureHarmonize built + runtime tested (AI crop 2x too sharp, auto-corrected). VacePrePassReference tail inputs removed (D-026). Aesthetic conditioning research: 20+ papers, WAN DiT injection points mapped. New workstream I: Texture Harmonize + Aesthetic Conditioning.
- **2026-04-10:** NV_FrequencyShapedNoise SHELVED after full debug cycle (D-029). Empirical evidence: flow-matching DiTs reject init-noise covariance priors — no usable operating window between "ignored" and "destabilizing". Knowledge captured in workstream I history. TextureHarmonize gained MAD stat mode for outlier-robust temporal stability (D-030).

## Archived Workstreams Index
<!-- Pointers to workstreams moved to ARCHIVE.md. -->

*No archived workstreams.*
