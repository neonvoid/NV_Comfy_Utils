# Status Board — NV_Comfy_Utils

> Auto-managed by `/handoff`. Content is never deleted — old entries move to ARCHIVE.md.
> Last updated: 2026-04-15

## Resume Context
<!-- Rewritten each `/handoff` run. What does a cold-start agent need RIGHT NOW? -->

- **Current focus:** Runtime test NV_SAM3Preprocess v1 on known SAM3 failure frames. Start conservative (clahe_clip_limit=1.5, everything else off). Escalate to gamma=1.3 + guided_filter_radius=3 if needed.
- **Critical files:** `src/KNF_Utils/sam3_preprocess.py` (new v1 node), `src/KNF_Utils/__init__.py` (registration updated), `src/KNF_Utils/guided_filter.py` (reused for denoise)
- **Known blockers:** None — file compiles, awaiting ComfyUI server restart for module reload.
- **Environment notes:** Gemini API quota exceeded (prior session) — multi-ai Gemini fell back to gemini-2.5-pro successfully. Earlier TextureHarmonize V2 runtime test from 2026-04-13 Resume Context still pending (workstream I untouched this session).

## Pulse
<!-- Last 2 session summaries, newest first. Older entries roll to Workstream Details. -->

### 2026-04-14/15 — Research arc: mask-domain editing → mocap+UE5 reframe → grade-the-plate SAM3 preprocess v1 shipped [coding + research]
- **Done:**
  - Researched mask-domain creative editing as control surface for VACE. Two parallel research agents concluded: nobody has shipped generalist mask-domain LoRA, every published mask-diffusion paper rebuilt the VAE, parametric body path is cheapest.
  - Three new research docs landed in the tree: `research/pipeline_techniques/2026-04-11_mask_domain_diffusion_feasibility.md`, `research/models_and_papers/2026-04-11_human_body_representation_models_taxonomy.md` (8-family taxonomy, current 2025-2026 SOTA), `research/models_and_papers/2026-04-12_image_editing_models_sota_update.md` (FLUX.2 dev/klein, HunyuanImage 3.0, Qwen-Image 2.0, BAGEL landscape).
  - Reframed approach after user disclosed **Vicon + UE5** access. Synthetic-render path (mocap → UE5 retarget → render alpha matte → VACE) dominates diffusion-based mask editing for body-swap use case. Saved as memory `mocap_and_game_engine_access.md`.
  - Clarified matchmove (camera) vs rotomation (subjects). Discussed Animate+VACE cascade architectures. Validated that iterative multi-pass mask editing ("fix character → fix trees → fix clouds") is production-viable with existing VAE drift fixes (CropColorFix V2 + NV_VaceLatentSplice + TextureHarmonize MAD). Bottleneck moved from VAE drift to mask quality.
  - TeleStyle deep dive (temporal propagation via positional encoding trick, NOT optical flow as secondary sources claimed). Current open-source image editing / gen SOTA survey.
  - Pivoted to **grade-the-plate SAM3 preprocessing** after user correctly diagnosed iterative SAM3 refinement as a dead end (same region → same embedding → same mask). Literature search found CLAHE evidence (Huang 2024, MedSAM: IoU +0.081, Dice +0.050) but RobustSAM showed preprocessing alone insufficient for hard VNS cases.
  - Multi-AI reviewed Phase 1 node design (Codex + Gemini). Implemented NV_SAM3Preprocess v1: guided filter denoise → gamma on luminance → CLAHE on Lab L (uint16-only L quantization, a/b stay float). Single output (not passthrough). Registered in `__init__.py`, passed `py_compile`.
- **Decisions:** Mocap+UE5 synthetic-render path dominates mask-domain diffusion for body-swap use cases (D-038). Animate+VACE cannot merge at model level — Replacement Mode may be one-shot solution (D-039). Multi-pass mask editing is production-viable; envelope ≤6 passes, ≥64px working-crop masks (D-040). SAM3 preprocessing = guided filter (not bilateral, avoids staircasing) + gamma on luma + Lab-L CLAHE, single output, targeted-fix not always-on (D-041).
- **Blockers:** None — code compiles, awaiting ComfyUI server restart + runtime test on known SAM3 failure frames.
- **Next:** Runtime test NV_SAM3Preprocess on dark-hair-on-dark-bg / skin-vs-warm-bg / semi-transparent-fabric cases. Start conservative (clahe_clip_limit=1.5 only). Escalate to gamma + guided filter if needed. If VNS cases still fail, next step is MatAnyone second-opinion segmentor.

### 2026-04-12b — TextureHarmonize multi-AI scope audit + 5 production fixes [coding + research]
- **Done:**
  - Runtime tested TextureHarmonize at multiple settings: highband_strength 0.5/1.0, context_scope full_frame/whole_crop/ring_only, shift 8.0/4.85.
  - Discovered full_frame scope gives OPPOSITE correction direction vs local scopes on DOF footage (1.43 vs 0.84 — background blur contaminates reference stats).
  - Confirmed highband reinjection causes ghost edges at denoise≥0.55 (spatial misalignment is fundamental, not tunable).
  - Discovered grain stage correctly does nothing — AI output is noisier than clean studio footage ("VAE fizz").
  - Multi-AI research (Codex + Gemini): both agreed on all 5 findings. Codex proposed grain_mode=match, Gemini named "VAE fizz" phenomenon.
  - Implemented 5 production fixes: default→whole_crop, highband max→0.5, denoise auto-taper (smoothstep zero above 0.6), grain_mode match (50% cap removal), full_frame DOF diagnostic warning.
- **Decisions:** Default context_scope=whole_crop not ring_only — 14x more pixels, same answer, better temporal stability (D-034). Highband max=0.5, auto-disabled above denoise 0.6 — spatial HF transfer unsafe at high denoise (D-035). Grain match mode — reduce "VAE fizz" when gen>ctx, capped 50% to prevent plastification (D-036). full_frame scope demoted — DOF-contaminated reference is wrong for subject compositing (D-037).
- **Blockers:** None — code complete, syntax verified, awaiting runtime test.
- **Next:** Runtime test with whole_crop default + grain_mode=match on the clothing swap shot. Compare with/without grain reduction visible. Test denoise auto-taper with connected denoise input.


## Active Workstreams

| ID | Name | Status | Last Touch | Key Context |
|----|------|--------|-----------|-------------|
| A  | Mask Pipeline UX Refactor | ACTIVE | 2026-04-06 | Renames done, deprecated removed, debug preview working. Audit doc complete. |
| B  | Edge-of-Frame Fix | STABLE | 2026-04-03 | Crop clamp + reflection/zeros padding — runtime tested, working |
| C  | Clothing/Bag Swap Pipeline | ACTIVE | 2026-04-08 | Full body + head swap. CropColorFix validation added (D-020). Multi-pass workflow stabilizing. |
| D  | Real-Time Mask Editor | STAGED | 2026-04-03 | Research complete — PySide6 + cached op graph MVP. Not started. |
| E  | Chunk Seam Continuity | ACTIVE | 2026-04-10 | NV_VaceLatentSplice built + runtime validated. Zero-drift tail overlap confirmed. |
| F  | Kling API Chunking | ACTIVE | 2026-04-09 | type="first_frame" hints on tail refs. Debug logging added. Awaiting runtime test of API acceptance. |
| G  | Masking & VFI Pipeline Research | ACTIVE | 2026-04-10 | Mocha for sub-object, SAM3 for full body, MatAnyone for edge refinement. GIMM-VFI stays. NV_MatchInterpFrames + RetimePrep/Restore manual fallback inputs. |
| H  | VACE Chunked Orchestrator | STAGED | 2026-04-10 | Architecture designed (4 multi-AI rounds). Single-queue-press chunked VACE inpainting with latent splice. |
| I  | Texture Harmonize + Aesthetic Conditioning | ACTIVE | 2026-04-12 | Multi-AI scope audit complete. 5 fixes: default→whole_crop, HB max→0.5 + denoise taper, grain match mode, full_frame DOF warning. Awaiting runtime test. |
| J  | SAM3 Input Quality (Grade-the-Plate) | ACTIVE | 2026-04-15 | NV_SAM3Preprocess v1 shipped (CLAHE + gamma + guided filter). Multi-AI reviewed. Awaiting runtime test. |

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
- TextureHarmonize per-stage estimator: std for Laplacian pyramid bands (edges ARE the signal), MAD for grain (edges are pollution). Using MAD for sharpness causes blurriness; using std for grain causes temporal strobing.
- High-band reinjection sigma controls the frequency split: too high → identity features leak from original into AI result (ghosting). 3.0 = safe default for 832×832.
- Highband reinjection (frequency separation) requires spatial alignment — unsafe above denoise ~0.5. Even at denoise 0.55, the sampler repositions fine detail enough to cause ghost edges from the original.
- full_frame context_scope includes background/OOF areas — gives opposite correction direction from local scopes on DOF footage. Use whole_crop or ring_only for subject compositing.
- AI diffusion output is often NOISIER than clean studio footage ("VAE fizz" — synthetic micro-noise from sampling + VAE decode). Grain synthesis add_only mode correctly does nothing; use match mode to dampen.
- SAM3 iterative refinement on the same failure region is a dead end — same input pixels produce the same encoder embedding and the same mask. To get new signal: (a) change the input via preprocessing (grade-the-plate — NV_SAM3Preprocess), (b) use a different segmentor with different failure modes (MatAnyone for hair, BiRefNet for soft edges), or (c) apply non-ML feature-aware morphology. Clicking "add more" on the same region does NOT create new signal.

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
| D-031 | 2026-04-12 | ACTIVE | I | Per-stage estimator: std for Laplacian pyramid sharpness (edges ARE the signal), MAD for grain residual (edges are pollution). Using wrong estimator for sharpness causes blurriness. |
| D-032 | 2026-04-12 | SUPERSEDED → D-035 | I | High-band reinjection (frequency separation) as primary fix for cross-model cartoon drift — grafts original HF onto AI LF inside mask |
| D-033 | 2026-04-12 | ACTIVE | I | Pre-process HF boost is dead end — model's prior actively ignores structured HF in reference latent during denoising |
| D-034 | 2026-04-12 | ACTIVE | I | Default context_scope=whole_crop — 14x more pixels than ring_only, same result, better temporal stability. ring_only for expert override only. |
| D-035 | 2026-04-12 | ACTIVE | I | Highband reinjection max=0.5, auto-disabled above denoise 0.6 via smoothstep taper. Spatial HF transfer fundamentally unsafe when sampler repositions fine detail. |
| D-036 | 2026-04-12 | ACTIVE | I | Grain match mode: reduce AI "VAE fizz" when gen_mad > ctx_mad, capped at 50% removal. Kills synthetic micro-noise without plastification. |
| D-037 | 2026-04-12 | ACTIVE | I | full_frame scope demoted — DOF/background contamination gives opposite correction direction. Only valid for flat content (anime/motion graphics). |
| D-038 | 2026-04-15 | ACTIVE | research | Mocap+UE5 synthetic-render path dominates mask-domain diffusion for body-swap / silhouette-edit use cases. Every published mask-diffusion paper rebuilt the VAE; parametric body (SMPL/MetaHuman in UE5) is cheaper, deterministic, ground truth. Mask-domain diffusion is Plan B. |
| D-039 | 2026-04-15 | ACTIVE | research | Animate+VACE cannot merge at model level (different model classes, no common weight structure). Use sequential cascade. Animate Replacement Mode (character_mask + background_video) may be one-shot solution for mocap-driven body swap — needs validation test. |
| D-040 | 2026-04-15 | ACTIVE | A,C,E,J | Iterative multi-pass mask editing is production-viable with existing VAE drift fixes (CropColorFix V2 + NV_VaceLatentSplice + TextureHarmonize MAD). Envelope: ≤6 passes comfortable, masks ≥64px in working-crop space. Mask quality is now dominant constraint, not VAE drift. |
| D-041 | 2026-04-15 | ACTIVE | J | NV_SAM3Preprocess v1: guided filter (not bilateral — avoids staircasing that SAM latches onto) + gamma on Rec.601 luminance + CLAHE on Lab L with uint16-only L quantization (a/b stay float). Single output (no passthrough — ComfyUI users branch IMAGE noodle). Targeted fix for degraded inputs, not always-on. |

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
**Current state:** ACTIVE — multi-AI scope audit complete, 5 production fixes applied, awaiting runtime test
**Goal:** Make AI-generated crops visually match source footage's texture quality (sharpness, grain, micro-contrast) via post-processing. For cross-model refinement (Kling→WAN), graft original's HF texture back onto identity-changed result.
**Key files:** `texture_harmonize.py` (production — 3 stages + denoise taper + grain match mode), `frequency_shaped_noise.py` (shelved), `multi_model_sampler.py` (has noise input — keep for future use)
**Active constraints:** VAE bottleneck prevents DiT-level grain/sharpness — post-decode is the ONLY viable control surface (D-029). Pre-process HF boost dead end (D-033). Per-stage estimator: std for pyramid, MAD for grain (D-031). Highband max=0.5 + denoise taper (D-035). Default scope=whole_crop (D-034). full_frame unsafe on DOF footage (D-037).

**Architecture (production):**
- Laplacian pyramid variance matching — sharpness + micro-contrast per frequency band
- Per-channel grain synthesis — add_only or match (reduces VAE fizz when gen>ctx, 50% cap). MAD-based stat.
- High-band reinjection — denoise-gated (smoothstep zero above 0.6), max 0.5. Low-denoise only.
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
- 2026-04-12 — TextureHarmonize major overhaul: per-stage stat split (std/MAD), context_scope modes, full_frame via stitcher (OOM-safe), high-band reinjection (Stage 3). Multi-AI reviewed (Codex). Foundations course created (5 docs).
- 2026-04-12 — Pre-process HF boost rejected as dead end (D-033). Post-process frequency separation confirmed as principled fix for cross-model cartoon drift.
- 2026-04-12 — Multi-AI scope audit: full_frame DOF contamination confirmed, 5 production fixes (D-034 through D-037). Grain match mode + denoise auto-taper.

**History:**
- **2026-04-10b | coding + research**
  Outcome: Built NV_TextureHarmonize (Laplacian pyramid + grain synthesis). Runtime tested (0.54/0.57 sharpness ratios). Removed VacePrePassReference tail inputs. 20+ papers reviewed. WAN DiT aesthetic injection points mapped.
  Decision: Tail inputs removed from PrePassReference (D-026). Texture harmonize post-decode only (D-027). Freq-shaped noise worth testing (D-028 PROVISIONAL).
  Next: Build NV_FrequencyShapedNoise.
- **2026-04-10 | coding + research (shelved experiment)**
  Outcome: NV_FrequencyShapedNoise built and tested through 2 multi-AI review rounds + bug fixes. Empirically confirmed dead end for flow-matching DiTs. TextureHarmonize gained MAD mode.
  Decision: SHELVE init-noise shaping for WAN/VACE (D-029). MAD as default texture statistic (D-030). Aesthetic control via post-decode only.
  Next: Build NV_VaceChunkedOrchestrator. Aesthetic problem is solved by TextureHarmonize.
- **2026-04-10c | coding + research**
  Outcome: Built NV_FrequencyShapedNoise (FFT→shaped noise), found 7 critical bugs, fixed all, still catastrophic rainbow artifacts. SHELVED — flow-matching DiTs reject init-noise covariance. TextureHarmonize gained MAD stat mode.
  Decision: SHELVE init-noise shaping for WAN/VACE (D-029). MAD as default texture stat (D-030).
  Next: Build orchestrator. Aesthetic problem owned by post-processing (TextureHarmonize).
- **2026-04-10/12 | coding + research**
  Outcome: TextureHarmonize overhaul — per-stage stat split (std for sharpness, MAD for grain), context_scope selector, full_frame via stitcher (OOM-safe), high-band reinjection for Kling→WAN cartoon drift. 5 Codex review fixes applied. Foundations course (5 docs) created.
  Decision: Per-stage estimator std/MAD (D-031). High-band reinjection post-process via frequency separation, not pre-process sharpening (D-032). Pre-process HF boost dead end — model's prior ignores structured HF input (D-033).
  Next: Runtime test highband_strength=0.5 on WAN-on-Kling identity refinement shot.

### J. SAM3 Input Quality (Grade-the-Plate)
**Current state:** ACTIVE — v1 shipped, syntax verified, awaiting runtime test
**Goal:** Improve SAM3 segmentation on degraded inputs (wispy hair on dark bg, skin-vs-warm-bg, semi-transparent fabric) via classical CV preprocessing — same VFX principle as keying: grade the plate, don't buy a better keyer.
**Key files:** `src/KNF_Utils/sam3_preprocess.py` (new), `src/KNF_Utils/guided_filter.py` (reused), `src/KNF_Utils/__init__.py` (registered)
**Active constraints:** Targeted fix for degraded inputs only — NOT always-on (gains small-to-negative on well-exposed photos). Per-frame CLAHE may flicker on video if clip_limit > 1.5. Does NOT solve true VNS cases (use MatAnyone/BiRefNet second-opinion for those). Evidence base narrow: Huang 2024 MedSAM is closest direct precedent; RobustSAM proves preprocessing alone insufficient for hardest cases.

**Architecture (v1):**
- Operation order (optional/skippable at each stage): guided filter denoise → gamma on Rec.601 luminance → CLAHE on Lab L channel
- Guided filter chosen over bilateral (both reviewers agreed: bilateral staircases, SAM latches onto staircases as false boundaries). Reuses existing `guided_filter.py` infrastructure, GPU-batched, auto-selects fast variant ≥720p.
- Gamma applied via per-pixel luminance ratio (preserves chroma) — >1.0 lifts shadows for dark-on-dark cases.
- CLAHE: float32 Lab throughout, only L quantized to uint16 [0, 65535] for OpenCV's CLAHE call. a/b never leave float. clipLimit capped at 3.0 (off-manifold above).
- Single output (no passthrough) — ComfyUI users branch the IMAGE noodle themselves for downstream compositing.

**Multi-AI review synthesis (Codex + Gemini, 2026-04-15):**
- Convergence: drop `apply_to` dropdown, drop `rgb_each`, drop sharpening/vibrance/ensemble from v1, add `gamma` on luminance, order = denoise→gamma→CLAHE, conservative defaults, quantize only L.
- Disagreement 1 (denoise algorithm): Gemini's guided-filter argument won over Codex's bilateral — staircasing is a hard SAM failure mode.
- Disagreement 2 (output shape): Gemini's single-output argument won over Codex's dual-output — passthrough is ComfyUI graph clutter.

**Parameters (all optional, default off except CLAHE):**
- `gamma` (0.5-2.0, default 1.0) — luminance shadow lift
- `clahe_clip_limit` (0.0-3.0, default 1.5) — 0 = disabled
- `clahe_tile_grid` (4-16, default 8) — OpenCV tileGridSize
- `guided_filter_radius` (0-16, default 0) — 0 = disabled
- `guided_filter_eps` (0.0001-0.1, default 0.01) — smoothing regularization

**Deferred to v2:**
- Temporal smoothing via EMA on per-tile L statistics (NOT first-frame LUT reuse — both reviewers rejected that as wrong trade-off)
- Ensemble output (union/intersection/voting across preprocessed variants)
- Presets (`wispy_hair`, `dark_on_dark`) — wait for runtime data before picking
- Kornia migration for GPU-native CLAHE if CPU per-frame becomes bottleneck

**Milestones:**
- 2026-04-15 — NV_SAM3Preprocess v1 built, multi-AI reviewed (Codex + Gemini), registered, syntax verified

**History:**
- **2026-04-14/15 | coding + research**
  Outcome: Built NV_SAM3Preprocess v1 — guided filter denoise → gamma on luminance → CLAHE on Lab L (uint16 L-only quantization, a/b stay float). Multi-AI review synthesized: drop bilateral for guided filter (no staircasing), drop dual output for single output, drop apply_to/rgb_each/sharpening, add gamma.
  Decision: Guided filter over bilateral, Lab-L CLAHE with L-only quantization, single output (D-041). Evidence: Huang 2024 (MedSAM: IoU +0.081), RobustSAM CVPR 2024 (preprocessing alone insufficient for VNS cases).
  Next: Runtime test on known failure frames — start with clahe_clip_limit=1.5 only.

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
- **2026-04-10/12:** NV_TextureHarmonize major overhaul: per-stage estimator split (std/MAD), context_scope modes (ring_only/whole_crop/full_frame), OOM-safe full-frame precompute via stitcher, high-band reinjection (Stage 3 frequency separation). Multi-AI research on edit model architectures + AI-on-AI cartoon drift diagnosis. Foundations course created (4 intuition modules + README, multi-AI reviewed). RetimeRestore manual fallback inputs added.
- **2026-04-12:** TextureHarmonize multi-AI scope audit: full_frame DOF contamination confirmed, 5 production fixes applied (scope default, HB cap+taper, grain match mode, DOF warning). Multi-AI consensus (Codex+Gemini) on all findings.
- **2026-04-14/15:** Mask-domain editing research arc → mocap+UE5 reframe → grade-the-plate SAM3 preprocess v1 shipped. Three new research docs (mask-domain diffusion feasibility, body representation taxonomy, image editing SOTA update). Vicon+UE5 access disclosed — synthetic-render path now preferred over diffusion-based mask editing for body-swap cases. Multi-pass mask editing validated as production-viable (≤6 passes, ≥64px masks). New workstream J: SAM3 Input Quality.

## Archived Workstreams Index
<!-- Pointers to workstreams moved to ARCHIVE.md. -->

*No archived workstreams.*
