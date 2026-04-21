# Status Board — NV_Comfy_Utils

> Auto-managed by `/handoff`. Content is never deleted — old entries move to ARCHIVE.md.
> Last updated: 2026-04-21

## Resume Context
<!-- Rewritten each `/handoff` run. What does a cold-start agent need RIGHT NOW? -->

- **Current focus:** 5-URL YouTube test batch is RUNNING NOW via new NV_GeminiYoutubeBatchExtractor (shell rendering / Houdini / UE course material). First real validation of the node + thinkingLevel="low" fix. Once complete, verify JSON names match YT titles on disk.
- **Recent workflow shift:** User is moving from OR base64 path (36-video local corpus from 04-18, still pending) → Gemini-direct YouTube path for new course material. No base64 ceiling, free input tokens, auto-discovered public videos via oEmbed titles.
- **Next dev focus:** Verify 5-URL batch JSON filenames + content. Then `/course-synth` skill invocation on the extracted JSONs. Then workstream H Phase 1 (plan_chunks math). J runtime test still pending post-SAM3-fix.
- **Critical files:** `src/KNF_Utils/gemini_video_course.py` (+698 lines this session, now ~3100 LOC), `src/KNF_Utils/prompt_refiner.py` (OR slug list unchanged), `web/clone_with_connections.js` (rewritten late-session for ComfyUI frontend 1.42.x — patches `_deserializeItems` instead of the old pointerdown + graph.add hook).
- **Known blockers:** None active — 5-URL batch runtime in-flight. 36-video OR corpus remains pending from 04-18 but is no longer prioritized.
- **Environment notes:** Gemini 3 thinkingLevel enum tightened — `"disabled"` no longer accepted on 3.1-pro-preview; `"low"` is canonical minimum. YouTube oEmbed (`https://www.youtube.com/oembed`) is no-auth, used for human-readable JSON filenames. Legacy `youtube_<id>.json` caches auto-migrate to `{title}_{id}.json` on first batch run. ComfyUI frontend is 1.42.10 — frontend extensions now target `_deserializeItems` as the sustainable chokepoint between Vue-nodes-mode and classic canvas modes (D-060).

## Pulse
<!-- Last 2 session summaries, newest first. Older entries roll to Workstream Details. -->

### 2026-04-21 (late session) — Fixed Shift+Alt+Drag clone-with-connections for ComfyUI frontend 1.42.x [coding + refactor]
- **Done:**
  - Diagnosed clone-with-links breakage after ComfyUI frontend upgrade (1.39.19 → 1.42.10). New frontend has TWO clone paths: Vue-nodes-mode calls `LGraphCanvas.cloneNodes` from a node DOM pointerdown handler; classic canvas mode calls `this._deserializeItems(this._serializeItems([t]))` directly from the LGraphCanvas pointer handler, gated by `!Z.vueNodesMode`. Old extension hooked pointerdown + graph.add — never fired under new dispatch.
  - First attempt patched `LGraphCanvas.cloneNodes` — caught Vue path but user's frontend runs classic mode, so patch never intercepted. Confirmed via `__nvCloneDiag()` + first-call `console.warn` diagnostics.
  - Final fix: patched `LGraphCanvas.prototype._deserializeItems` (chokepoint for BOTH paths). Leverages native `connectInputs: true` option + flipping `LiteGraph.ctrl_shift_v_paste_connect_unselected_outputs` setting while shift is held. No manual `sourceNode.connect()` calls — restored connections live inside the same `beforeChange/afterChange` block as the clone, so single Ctrl+Z fully undoes.
  - Hardening: diagnostics use `console.warn` (punches through devtools log-level filter that hides `console.log` — "X hidden" counter was the smoking gun), pointerdown resync for shiftHeld (focused inputs can swallow keyup), `__nvCloneDiag()` global for live state inspection, graceful-disable warning with actionable hint when upstream API shape changes.
  - Runtime tested — works end-to-end on classic canvas mode.
- **Decisions:** D-060 (sustainable upstream-API patching pattern — target highest-level semantic chokepoint + leverage native flags, don't re-implement behavior manually).
- **Blockers:** none.
- **Next:** None for this thread — hand back to the YT batch work (5-URL test runtime in-flight, see morning entry).

### 2026-04-21 — NV_GeminiYoutubeBatchExtractor built; title-based JSON names; Gemini 3 thinkingLevel "disabled" bug fixed [coding + bugfix]
- **Done:**
  - **NV_GeminiYoutubeBatchExtractor** (~400 LOC) — Gemini-direct YouTube batching. Multiline URL input, `#` comments + blanks skipped, dedup by canonical URL. OR models raise up-front (OR can't ingest YT bytes). Dry-run previews titles + cache status + est. cost (output-only per YT preview pricing). Schema-versioned cache, partial-on-failure, Cancel-responsive. Reuses `_EXTRACTION_PROMPT` + `_generate_content` from single extractor.
  - **Title-based JSON filenames** (applies to both single extractor + batch): `{sanitized_title}_{video_id}.json`. Public oEmbed (`https://www.youtube.com/oembed`) for no-auth title fetch. Sanitizer keeps Unicode (emoji/CJK), strips Windows-illegal chars, 120-char cap. ID suffix guarantees uniqueness. `_source` gains `youtube_video_id`/`youtube_title`/`youtube_author`.
  - **Legacy-cache migration**: `_migrate_legacy_youtube_cache()` renames existing `youtube_<id>.json` → new title-based name on first run. Idempotent, runs pre-cache-check so prior extractions reuse instead of re-running.
  - **Gemini 3 thinkingLevel bug (runtime test exposed)**: `gemini-3.1-pro-preview` rejects `thinkingLevel: "disabled"` with INVALID_ARGUMENT. Worked on earlier 3.x preview checkpoints; enum has tightened. Changed to `"low"` — documented minimum, accepted across 3.x variants, same intent (strip thought tokens from json_mode output).
  - **400 error-handler masking real bug**: prior handler remapped every 400 to "private video" — hid the thinkingLevel config error behind misleading "likely private/unlisted" text. Tightened both batch + single to keyword-match on body (`private`, `unlisted`, `age`, `region`, etc.) before remapping. Generic 400s now surface their real message.
  - **Stale log string**: `[Gemini] Calling … thinking=disabled` was hardcoded; now reads `thinkingLevel` from payload dynamically. Will track future changes without code drift.
- **Decisions:** D-056 (YT batch node architecture — Gemini-direct only, multiline input, oEmbed for titles), D-057 (YT JSON filename scheme `{title}_{id}.json` + legacy migration), D-058 (Gemini 3 `thinkingLevel "low"` for json_mode supersedes `"disabled"` — enum rejected on 3.1), D-059 (400 error remapping must keyword-match body, not just status code).
- **Blockers:** 5-URL YT test batch is currently running (first real validation of the new node + thinkingLevel fix). 36-video OR corpus from 04-18 still pending and is now secondary — user's workflow is shifting toward Gemini-direct YT path.
- **Next:** Confirm 5-URL batch completes cleanly. Verify JSON filenames match YT titles. Verify legacy-migration path on a directory with pre-existing `youtube_<id>.json` files. Then the user's Pipeline A/B next steps (course-synth skill on the JSONs folder, workstream H Phase 1).

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
| H  | VACE Chunked Orchestrator | STAGED | 2026-04-15 | Architecture designed (4 multi-AI rounds). Deep mental model established with 600-frame walkthrough. Phase 1 = plan_chunks() math helper first. |
| I  | Texture Harmonize + Aesthetic Conditioning | ACTIVE | 2026-04-12 | Multi-AI scope audit complete. 5 fixes: default→whole_crop, HB max→0.5 + denoise taper, grain match mode, full_frame DOF warning. Awaiting runtime test. |
| J  | SAM3 Input Quality (Grade-the-Plate) | ACTIVE | 2026-04-17 | NV_SAM3Preprocess v1 + SAM3VideoSegmentation auto-detect refactor (A') done. Awaiting runtime test. |
| K  | Gemini Course Extractor / OpenRouter | ACTIVE | 2026-04-21 | NV_GeminiYoutubeBatchExtractor shipped (Gemini-direct YT batch, oEmbed titles → `{title}_{id}.json` with legacy migration). Gemini 3.x `thinkingLevel` "disabled" → "low" (enum tightened on 3.1-pro-preview). 5-URL batch runtime in-flight. |

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
- OpenRouter does NOT publish `google/gemini-3-pro-preview` — use `google/gemini-3.1-pro-preview`, `google/gemini-3-flash-preview`, or `google/gemini-3.1-flash-lite-preview`. Authoritative list: `src/KNF_Utils/prompt_refiner.py:455-457`.
- ComfyUI dynamic widget hide/show is a serialization hazard. If a node needs mode-dependent inputs, prefer **input-presence auto-detection** (Option A' pattern in SAM3VideoSegmentation) over JS hide/show. Splicing `node.widgets` corrupts `widgets_values` on reload; even the correct type-swap pattern accumulates risk.
- WAN VAE `Encoder3d` / `Decoder3d` API changed (2026-04-ish core): encoder uses 2-frame chunks with `final=` flag and may return None for intermediate chunks; decoder now returns a list of output chunks (not a tensor); `count_cache_layers()` replaces `count_conv3d()` for feat_map sizing. Any streaming/chunked VAE helper must mirror native `_encode` / `_decode` exactly or break silently. See `streaming_vace_to_video.py` for the canonical helper.
- OR `/api/v1/generation/{id}` is eventually-consistent (1-3s lag) — must poll with exponential backoff. Audit lookup must be best-effort, NEVER block extraction success. Returns `or_generation_lookup_status: "hit"|"miss"|"skipped_no_id"` for diagnostics.
- OR settled `total_cost == 0` is VALID (cached/free generation). Gate with `>= 0`, NOT `> 0`. Otherwise legitimate free calls keep stale inline estimates and cost accounting drifts upward.
- `_parse_extraction_json` returns dict today, but ANY mutation of `extraction[…]` MUST be guarded by `isinstance(extraction, dict)` first. A future refactor returning a string/list crashes the entire batch mid-loop with `TypeError`. Pattern: wrap non-dict in `{"raw_output": extraction, "parse_error": "…"}` before mutating.
- Schema/prompt changes to `_EXTRACTION_PROMPT` or `_EXTRACTION_SCHEMA` MUST bump `_EXTRACTION_SCHEMA_VERSION`. Otherwise cached extractions silently shadow the new shape and downstream consumers receive stale-shape data.
- OR credit pre-flight is informational only — NEVER hard-raise on insufficient balance. Prompt caching + upstream discounts make pre-estimates unreliable; OR's 402 on the actual call is the real failure signal. Pattern: warn loudly + proceed.
- Truncated/malformed extractions MUST go to `<video>_partial.json` (not main `<video>.json`) so the next run retries instead of returning broken cached data. Clean up stale partials on later success.
- `GOP=1` at low fps DESTROYS compression for static content — every frame is a standalone keyframe, killing inter-frame delta. Screen-recording transcode MUST use GOP=60+ to exploit static-content redundancy. libx264 default (250) also works. This was the OR transcode bug shipped 2026-04-18.
- OpenRouter upstream gateways (Vertex etc.) return **502 Bad Gateway** on oversized payloads, NOT 413. The 502 is upstream timeout mid-upload-body — retries don't help for size issues; they help only for transient errors. Use predictive size gate to avoid the round-trip.
- `subprocess.run()` on Windows for ffmpeg/ffprobe REQUIRES `encoding="utf-8", errors="replace"` (NOT `text=True`). Default cp1252 codec crashes reader threads on non-ASCII filenames or progress output. Applies to all 3 subprocess sites in `gemini_video_course.py`.
- OR base64 transport caps at ~64MB encoded (~48MB raw) before gateway timeouts. Audio track at 32kbps mono bottlenecks OR-viable video length at ~75min — no amount of compression frees bytes beyond that since speech intelligibility has a lower bound.
- ComfyUI Cancel button (queue widget X icon) requires `comfy.model_management.throw_exception_if_processing_interrupted()` calls at natural pause points to be responsive. Otherwise cancel waits for the current Python frame to return — could be 10+ min on a mid-API-call. Pattern: call at loop tops + chunked sleep steps.
- Gemini 3.x `thinkingLevel` enum: `"disabled"` is REJECTED on gemini-3.1-pro-preview with INVALID_ARGUMENT (verified 2026-04-21). Valid values: `low`/`medium`/`high`. Use `"low"` to minimize thinking cost in json_mode without hitting enum errors (thoughts are stripped from JSON output anyway). `thinkingBudget: 0` was not tested as a fallback — `"low"` was sufficient.
- HTTP status-code-only error remapping is dangerous — always keyword-match the response body before suggesting causes. A generic config 400 (bad enum value, bad payload) and a privacy-driven 400 look identical at the status level; body keyword matching is mandatory. Applies equally to Gemini direct, OpenRouter, and any future transport.
- YouTube oEmbed (`https://www.youtube.com/oembed?url=<canonical>&format=json`) is the canonical no-auth title lookup. Returns `{title, author_name, thumbnail_url, ...}`. Soft-fail on any error (private videos, network timeouts, geo-blocks) — never raise in code paths that also make a paid API call downstream.
- ComfyUI frontend 1.42.x has DUAL clone paths: Vue-nodes-mode calls `LGraphCanvas.cloneNodes` from a node DOM pointerdown handler; classic canvas mode calls `this._deserializeItems(this._serializeItems([t]))` directly from the LGraphCanvas pointer handler (gated by `!Z.vueNodesMode`). Patching only `cloneNodes` misses classic-mode users. Both paths converge at `LGraphCanvas.prototype._deserializeItems` — that is the sustainable chokepoint for clone-related extensions.
- LiteGraph `_deserializeItems` supports native external-link restoration via `connectInputs: true` option, gated behind `LiteGraph.ctrl_shift_v_paste_connect_unselected_outputs` setting. BOTH must be on; deserializer falls back to `graph.getNodeById()` for origin_ids outside the cloned selection. Enables clone/paste with external inputs without manual `connect()` calls, and keeps the restoration inside the native `beforeChange/afterChange` block (single Ctrl+Z).
- Chrome/Edge devtools "X hidden" counter = default log-level filter hides `console.log`. Frontend diagnostic messages that must survive filtering (load confirmations, first-call traces, restore counts) should use `console.warn`.

## Project Decisions Index
<!-- Numbered decisions with lifecycle status. Never renumber IDs. -->

| ID | Date | Status | Workstream | Description |
|----|------|--------|-----------|-------------|
| D-001 | 2026-04-01 | ACTIVE | A | Prefixed naming convention: `cleanup_*`, `crop_*`, `vace_*` (not double-underscore) |
| D-002 | 2026-04-01 | ACTIVE | A | One config node with categorized params (not split into multiple nodes, not modes/presets) |
| D-003 | 2026-04-01 | SUPERSEDED → D-006 | A | → ARCHIVE.md |
| D-004 | 2026-04-01 | ACTIVE | B | Crop clamped to frame bounds (no canvas extension). Isotropic shrink preserves aspect ratio. |
| D-005 | 2026-04-01 | ACTIVE | B | Stitch inverse warp uses `zeros` padding (not reflection) — compositing is mask-gated so zeros are invisible |
| D-006 | 2026-04-02 | ACTIVE | A | Remove all deprecated params — no backward compat for old workflows. Clean UI. |
| D-007 | 2026-04-02 | ACTIVE | A | `forceInput:True` does NOT preserve old widget values — ComfyUI treats as connection-only inputs |
| D-008 | 2026-04-03 | ACTIVE | C | `boundary_lock` restores original pixels via `torch.where`, doesn't darken — exposes what multiband created |
| D-009 | 2026-04-03 | ACTIVE | C | Use `crop_expand_px` for spatial coverage, NOT CropColorFix `blend_expansion` (causes dark borders) |
| D-010 | 2026-04-06 | ACTIVE | E | Skip identity_anchor until color-normalized — raw Kling brightness differs from CropColorFix'd VACE output |
| D-011 | 2026-04-06 | ACTIVE | E | Committed noise (Jan 2026) empirically failed for fine-detail drift; `reference_latents` dead code for WAN |
| D-012 | 2026-04-06 | SUPERSEDED → D-016 | E | → ARCHIVE.md |
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
| D-028 | 2026-04-10 | SUPERSEDED → D-029 | I | → ARCHIVE.md |
| D-029 | 2026-04-10 | ACTIVE | I | NV_FrequencyShapedNoise SHELVED — flow-matching DiTs read init-noise covariance changes as content, not as priors. No usable operating window between "ignored" and "destabilizing". Post-processing (TextureHarmonize) is the right architectural answer. |
| D-030 | 2026-04-10 | ACTIVE | I | TextureHarmonize defaults to MAD (Median Absolute Deviation) instead of std for texture spread — robust to single-pixel outliers (eyelashes/highlights) that caused temporal strobing in tight crops as masks wiggled |
| D-031 | 2026-04-12 | ACTIVE | I | Per-stage estimator: std for Laplacian pyramid sharpness (edges ARE the signal), MAD for grain residual (edges are pollution). Using wrong estimator for sharpness causes blurriness. |
| D-032 | 2026-04-12 | SUPERSEDED → D-035 | I | → ARCHIVE.md |
| D-033 | 2026-04-12 | ACTIVE | I | Pre-process HF boost is dead end — model's prior actively ignores structured HF in reference latent during denoising |
| D-034 | 2026-04-12 | ACTIVE | I | Default context_scope=whole_crop — 14x more pixels than ring_only, same result, better temporal stability. ring_only for expert override only. |
| D-035 | 2026-04-12 | ACTIVE | I | Highband reinjection max=0.5, auto-disabled above denoise 0.6 via smoothstep taper. Spatial HF transfer fundamentally unsafe when sampler repositions fine detail. |
| D-036 | 2026-04-12 | ACTIVE | I | Grain match mode: reduce AI "VAE fizz" when gen_mad > ctx_mad, capped at 50% removal. Kills synthetic micro-noise without plastification. |
| D-037 | 2026-04-12 | ACTIVE | I | full_frame scope demoted — DOF/background contamination gives opposite correction direction. Only valid for flat content (anime/motion graphics). |
| D-038 | 2026-04-15 | ACTIVE | research | Mocap+UE5 synthetic-render path dominates mask-domain diffusion for body-swap / silhouette-edit use cases. Every published mask-diffusion paper rebuilt the VAE; parametric body (SMPL/MetaHuman in UE5) is cheaper, deterministic, ground truth. Mask-domain diffusion is Plan B. |
| D-039 | 2026-04-15 | ACTIVE | research | Animate+VACE cannot merge at model level (different model classes, no common weight structure). Use sequential cascade. Animate Replacement Mode (character_mask + background_video) may be one-shot solution for mocap-driven body swap — needs validation test. |
| D-040 | 2026-04-15 | ACTIVE | A,C,E,J | Iterative multi-pass mask editing is production-viable with existing VAE drift fixes (CropColorFix V2 + NV_VaceLatentSplice + TextureHarmonize MAD). Envelope: ≤6 passes comfortable, masks ≥64px in working-crop space. Mask quality is now dominant constraint, not VAE drift. |
| D-041 | 2026-04-15 | ACTIVE | J | NV_SAM3Preprocess v1: guided filter (not bilateral — avoids staircasing that SAM latches onto) + gamma on Rec.601 luminance + CLAHE on Lab L with uint16-only L quantization (a/b stay float). Single output (no passthrough — ComfyUI users branch IMAGE noodle). Targeted fix for degraded inputs, not always-on. |
| D-042 | 2026-04-15 | PROVISIONAL | tooling | Excalidraw selected for repo-committed pipeline + workstream visual tracking. VS Code extension, JSON files tracked in git, lives next to code (not Miro/Notion/GitHub Projects). `node_notes/diagrams/` is canonical location. |
| D-043 | 2026-04-15 | PROVISIONAL | H | Orchestrator Phase 1 must land `plan_chunks()` math helper with unit tests BEFORE sampler integration. Frame-count math is the gate — if rollback/edge-case logic is wrong, the whole loop produces misaligned garbage. Validate against 600-frame worked example. |
| D-044 | 2026-04-17 | ACTIVE | K,tooling | Cross-repo OpenRouter Gemini slug alignment. All dropdowns (gemini_video_course.py, nodes.py, AVM vlm_sam3_bridge.py) use the verified list from prompt_refiner.py. |
| D-045 | 2026-04-17 | SUPERSEDED → D-047 | tooling | → ARCHIVE.md |
| D-046 | 2026-04-17 | ACTIVE | tooling | Streaming WAN VAE helpers must mirror native `_encode`/`_decode` exactly: 2-frame chunks, `final=(i==iter_-1)` flag, None-skip for intermediate encoder chunks, list-of-tensors for decoder output, `count_cache_layers()` for feat_map. API drifts silently — test after every ComfyUI core pull. |
| D-047 | 2026-04-17 | ACTIVE | J,tooling | SAM3VideoSegmentation uses input-presence auto-detection instead of a `prompt_mode` dropdown. Mutual exclusion of text+points raises ValueError. Supersedes D-045 (widget-hide patch) — structural fix over JS patch. Debated with multi-AI (Codex+Gemini converged on A'). |
| D-048 | 2026-04-18 | ACTIVE | K | Partial-on-failure cache pattern. Truncated/malformed extractions write to `<video>_partial.json` (sibling), NEVER to main `<video>.json`. Schema_version cache-invalidates on `_EXTRACTION_SCHEMA_VERSION` bumps. Stale partials cleaned on later success. |
| D-048a | 2026-04-18 | ACTIVE | K | Defensive `isinstance(extraction, dict)` wrap before mutating `extraction[_source]` — `_parse_extraction_json` returns dict today, but a refactor returning non-dict would crash the whole batch mid-run. Caught in multi-AI review. |
| D-049 | 2026-04-18 | ACTIVE | K | Auto-token formula: `int((1024 + duration_min × 280) × 1.25)` clamped to `[floor, 65536]`. Empirical 280 tok/min for curriculum schema (1 segment per 1-2 min × 6-8 fields × ~30 tok/field + 25% safety margin). 90-min video budgets ~33K tokens. |
| D-050 | 2026-04-18 | ACTIVE | K | OR `/api/v1/generation/{id}` post-call lookup is cost ground truth. Settled `total_cost >= 0` (including 0 for cached/free) overrides inline `usage.cost`. Inline preserved as `inline_cost_usd` for comparison. Best-effort polling 1.5→12s exponential backoff. |
| D-051 | 2026-04-18 | ACTIVE | K | OR credit pre-flight is warn-only, NEVER raises. Prompt caching + upstream discounts make pre-estimates unreliable; OR 402 on the actual call is the real signal. `_or_check_credits()` returns structured `{ok, kind, status_code, detail}` distinguishing 401/403/500/timeout. |
| D-052 | 2026-04-18 | ACTIVE | K,course | Course pipeline production path = extract (Comfy node) → folder of JSONs → `/course-synth` skill (5-stage stateful with golden-sample voice calibration + 4-critic adversarial QA). Synth NODE is one-shot smoke test only — useful for coherence check + model A/B, not deliverable production. |
| D-053 | 2026-04-18 | ACTIVE | K | OR transcode profile rewritten: 1fps/GOP=60/720p/CRF30/32kbps mono audio. GOP=1 was the bug — disabled all inter-frame compression at low fps, making files grow or shrink only marginally. GOP=60 lets libx264 exploit screen-content static redundancy. Audio track caps OR-viable video length at ~75min. |
| D-054 | 2026-04-18 | ACTIVE | K | OR pre-flight hard-fails entire batch if any video would exceed 48MB raw ceiling after aggressive transcode. Refuses to start instead of silently failing per-video mid-run. Error message lists all offenders + 4 alternatives (direct Gemini / video_url / pre-split / future split-pipeline). Cached videos excluded from check. |
| D-055 | 2026-04-18 | ACTIVE | K | ComfyUI Cancel button (queue widget X icon) responsive during batch via `throw_exception_if_processing_interrupted()`. Inserted at top of each video iteration + OR `/generation` polling backoff sleeps (chunked 0.5s steps). Cancel latency: ~30-60s (finishes current API call), not instant. Safe ImportError fallback no-op if Comfy not in path. |
| D-056 | 2026-04-21 | ACTIVE | K | NV_GeminiYoutubeBatchExtractor: Gemini-direct only (OR raises up-front). Multiline URL input with `#` comments, dedup by canonical URL. oEmbed for titles (public, no auth, soft-fail). Separate from folder batch because inputs + transport + pricing model differ enough. |
| D-057 | 2026-04-21 | ACTIVE | K | YouTube JSON filename scheme `{sanitized_title}_{video_id}.json`. Sanitizer preserves Unicode, strips Windows-illegal, 120-char cap. ID suffix = collision-free. Legacy `youtube_<id>.json` migrated in-place pre-cache-check. Applies to both single + batch extractor. |
| D-058 | 2026-04-21 | ACTIVE | K | Gemini 3.x `thinkingLevel` enum tightened — `gemini-3.1-pro-preview` rejects `"disabled"` as INVALID_ARGUMENT. Use `"low"` for json_mode: documented minimum, accepted across 3.x, same cost-saving intent (thoughts stripped from JSON output anyway). Supersedes prior implicit assumption that `"disabled"` worked. |
| D-059 | 2026-04-21 | ACTIVE | K | HTTP status-code-only error remapping masks real causes. 400 handlers (batch + single) now keyword-match response body (`private`/`unlisted`/`age`/`region`/etc.) before suggesting YouTube privacy issue. Generic config 400s now surface the raw API message. |
| D-060 | 2026-04-21 | ACTIVE | tooling,frontend | Sustainable frontend upstream-API monkey-patches target the highest-level semantic chokepoint (e.g. `_deserializeItems`, NOT `graph.add` or pointerdown) and leverage native options/flags where available (`connectInputs`, `ctrl_shift_v_paste_connect_unselected_outputs`). Inherits native undo batching (single Ctrl+Z), avoids event-dispatch fragility across frontend refactors, shrinks surface area. Clone-with-connections extension rewritten under this pattern for ComfyUI 1.42.x. |

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
- **2026-04-17 | coding + refactor (cross-repo)**
  Outcome: WAN VAE streaming helpers rewritten against current core API — encoder uses 2-frame chunks with `final=` flag (None-returning intermediates), decoder returns list-of-chunks (not tensor), `count_cache_layers()` replaces `count_conv3d()`. Three files touched. Multi-AI reviewers caught a CPU-spillover regression; fixed to keep input on CPU, slice per-iter. Unblocks VacePrePassReference + other streaming-VAE callers.
  Decision: Streaming VAE helpers MUST mirror native `_encode`/`_decode` exactly (D-046). API drift surface — test after every ComfyUI core pull.
  Next: Runtime test VacePrePassReference 40-frame encode after restart.

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
- **2026-04-15 | research + tooling**
  Outcome: Orchestrator mental model established via 600-frame walkthrough (three-prepend systems, two-rails tail continuity, last-chunk rollback math). Excalidraw visual tracking infrastructure shipped to node_notes/diagrams/.
  Decision: Excalidraw for repo-committed visualization (D-042). Phase 1 gate: plan_chunks() math with unit tests before sampler integration (D-043).
  Next: Phase 1 scaffold — plan_chunks(total, chunk_size, tail_overlap) with tests (4k+1, rollback, edge cases).

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
- **2026-04-12 | coding + research**
  Outcome: Multi-AI scope audit on TextureHarmonize identified 5 production fixes: default→whole_crop, highband max=0.5 + denoise auto-taper, grain match mode (anti-"VAE fizz"), full_frame DOF warning. Implemented + syntax verified.
  Decision: whole_crop default (D-034), highband capped + tapered (D-035), grain match mode (D-036), full_frame demoted (D-037).
  Next: Runtime test with whole_crop + grain_mode=match on clothing swap shot.

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
- **2026-04-17 | refactor (cross-repo)**
  Outcome: SAM3VideoSegmentation widget-scramble bug solved structurally via Option A' (multi-AI debate converged). prompt_mode dropdown removed, mode inferred from input presence. Dynamic JS extension deleted.
  Decision: Auto-detect mode supersedes prompt_mode dropdown (D-047, supersedes D-045).
  Next: Runtime test NV_SAM3Preprocess pipeline end-to-end with fixed SAM3VideoSegmentation.

### K. Gemini Course Extractor / OpenRouter
**Current state:** ACTIVE — NV_GeminiYoutubeBatchExtractor shipped 2026-04-21. Title-based JSON names + legacy migration live for both single and batch extractors. Gemini 3.x thinkingLevel bug (enum tightened on 3.1-pro-preview) fixed in-session. 5-URL YT batch runtime in-flight. 36-video OR corpus from 04-18 still pending (deprioritized).
**Goal:** Multi-provider video extraction for tutorial course analysis — OpenRouter escape valve for Gemini rate limits, expanded model catalog for cost/quality comparison, length-safe + audit-rich. 2026-04-21 adds Gemini-direct YouTube batching as a parallel transport for online course material.
**Key files:** `src/KNF_Utils/gemini_video_course.py` (~3100 lines), `src/KNF_Utils/api_keys.py`
**Active constraints:** Non-Gemini models drop audio (D-052 → use Gemini routes for Chinese voiceover). OR video = base64 inline (~200MB+ for 90-min videos at 1fps could trigger 413 — flagged by reviewers but not yet hit). Auto-optimize transcodes to 1fps. Length-safe via `_check_extraction_health()` + partial-on-failure cache. Schema-version cache invalidation. OR settled cost is ground truth via `/generation/{id}`. Credit pre-flight warn-only. OR cannot ingest YouTube URLs — YT batch node raises up-front. Gemini 3.x `thinkingLevel` must be `low`/`medium`/`high` (not `disabled`). 400 error remapping must keyword-match body before suggesting privacy causes.

**Architecture (production):**
- Two extractors: single-video `NV_GeminiVideoExtractor` + batch `NV_GeminiBatchExtractor`
- Two transports: direct Gemini Files API (upload→poll→generate→delete) OR OpenRouter (optional ffmpeg transcode → base64 inline → /generation lookup)
- Per-video output budget scales with duration via `_scale_max_tokens_by_duration()` (default `auto_by_duration` mode, `manual` legacy fallback)
- Cache validated by model + media_resolution + video_path + schema_version; broken extractions go to `_partial.json` not main path
- `_source` block records: schema_version, max_tokens_mode/effective, parse_status, finish_reason, OR provider/native tokens/cache discount/timing/settled cost

**Milestones:**
- 2026-04-15 — OR mode + auto media optimization + 13-model catalog shipped. Multi-AI reviewed.
- 2026-04-17 — Slug alignment across 3 repos (D-044).
- 2026-04-18 — Phase 2 length safety (D-048, D-048a) + auto-token scaling (D-049) + OR audit endpoints (D-050, D-051) + course pipeline workflow clarified (D-052). Multi-AI review caught 7 bugs all fixed in same session.
- 2026-04-18 (pt 2) — Runtime validation surfaced OR transcode bug (GOP=1 killed compression). In-session fix: rewrote profile to 720p/CRF30/32kbps mono (predicts all 36 corpus videos fit under 48MB raw). Added predictive gate, pre-flight hard-fail (D-054), ComfyUI Cancel button wiring (D-055), Windows subprocess UTF-8 fix.
- 2026-04-21 — NV_GeminiYoutubeBatchExtractor shipped (~400 LOC). Title-based JSON names via oEmbed + legacy `youtube_<id>.json` migration (D-056/057). Runtime-exposed + same-session fixed: Gemini 3 thinkingLevel `"disabled"` INVALID_ARGUMENT on 3.1-pro-preview → `"low"` (D-058). 400 error-handler tightened to keyword-match body; had been masking the thinkingLevel config error behind "likely private video" text (D-059).

**History:**
- **2026-04-15 | coding**
  Outcome: Full OR path on both extractors. Auto ffmpeg optimization (1fps/CRF26/≤1080p/80kbps AAC). _OR_PRICING from live API. 13 OR routes. provider.order pins Vertex for google/*. json_schema fallback. 300s timeout. Cache to temp/nv_video_cache/.
  Decision: FPS reduction (not bitrate) for screen recording optimization. Pin Vertex backend for google/* (AI Studio requires YouTube URLs). data_collection="deny" on all OR requests.
  Next: Runtime test on 36-video VFX tutorial corpus. Cost differential analysis across models.
- **2026-04-17 | coding + triage**
  Outcome: Fixed invalid `google/gemini-3-pro-preview` slug. Aligned with authoritative prompt_refiner.py list across 3 repos. Cross-repo: AVM gained OpenRouter provider.
  Decision: Canonical slug list enforced across repos (D-044).
  Next: Runtime test on 36-video VFX corpus unchanged.
- **2026-04-18 | coding + refactor**
  Outcome: Three layered hardening passes — Phase 2 length safety (`_check_extraction_health()`, schema_version cache, partial-on-failure caching, `finish_reason` capture across both transports), auto-token scaling (~280 tok/min × 1.25 safety margin, `max_tokens_mode` dropdown, per-video `out_max=` in dry-run), OR audit endpoints (`/generation/{id}` for settled cost + provider attribution + native tokens, `/credits` warn-only pre-flight). Multi-AI review caught 7 bugs all fixed in same session: settled_cost ≥ 0 (free generations valid), defensive `isinstance(dict)` before mutation, soft-warn vs hard-raise on credit shortfall, stale partial cleanup, structured credits errors, `or_generation_lookup_status` surfaced, batch summary splits succeeded/failed cost.
  Decision: Partial-on-failure cache (D-048), defensive isinstance wrap (D-048a), 280-tok/min auto-token formula (D-049), settled `total_cost >= 0` overrides inline (D-050), credit pre-flight is warn-only (D-051), course production path = extract→SKILL not extract→synth-node (D-052).
  Next: ComfyUI restart + dry-run on 36-video corpus. Then 2-video Flash vs Pro calibration (~$0.50). Then commit full batch with chosen model.
- **2026-04-18 (pt 2) | coding + runtime test**
  Outcome: Discovered OR GOP=1 transcode bug via real run — files didn't shrink (some grew 6-9%), 24+ videos failed on 64MB ceiling with 502 gateway timeouts. Rewrote profile (1fps/GOP=60/720p/CRF30/32kbps mono), added predictive size gate, pre-flight oversized check that hard-fails whole batch, ComfyUI Cancel button wiring, Windows subprocess UTF-8 encoding fix (3 sites).
  Decision: GOP=60 for screen content (D-053 supersedes implicit GOP=1 default), hard-fail pre-flight > per-video silent failures (D-054), Cancel button via `throw_exception_if_processing_interrupted` at iteration tops + polling backoff chunks (D-055).
  Next: ComfyUI restart. Re-run batch — 14 cached reuse, 22 retry with new profile. Then /course-synth skill invocation on extracted JSONs.
- **2026-04-17 | coding**
  Outcome: Fixed invalid `google/gemini-3-pro-preview` OR slug in gemini_video_course.py; aligned with authoritative list in prompt_refiner.py across 3 repos.
  Decision: Canonical slug list enforced across repos (D-044).
  Next: Runtime test of 36-video VFX corpus unchanged.
- **2026-04-21 | coding + bugfix**
  Outcome: NV_GeminiYoutubeBatchExtractor shipped — Gemini-direct YT batching with multiline URL input, oEmbed titles, legacy cache migration, dry-run preview with titles. Title-based `{sanitized_title}_{video_id}.json` naming applied to both single + batch extractors. Runtime-exposed: Gemini 3.x `thinkingLevel "disabled"` rejected on 3.1-pro-preview (enum tightened) — fixed to `"low"`. 400 error-handler had been masking real config errors behind "private video" text — tightened to keyword-match body. Stale `thinking=disabled` log string fixed to read payload dynamically.
  Decision: YT batch node Gemini-direct only (D-056), JSON filename scheme `{title}_{id}.json` + legacy migration (D-057), thinkingLevel `"low"` supersedes `"disabled"` for json_mode (D-058), 400 remapping must keyword-match body (D-059).
  Next: 5-URL test batch currently running — verify filenames match YT titles, verify legacy-migration path. Then `/course-synth` skill invocation on JSONs. Then H Phase 1 (plan_chunks math).

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
- **2026-04-15:** NV_GeminiVideoExtractor / NV_GeminiBatchExtractor gain OpenRouter mode with auto media optimization (1fps/CRF26 ffmpeg, tuned for screen-recording tutorials). 13 OR video-capable model routes added with live pricing. New workstream K: Gemini Course Extractor / OpenRouter.
- **2026-04-15:** Orchestrator (workstream H) deep mental model established — 600-frame fight video walkthrough documents chunk planning, two-rails-for-tail-continuity intuition, three-prepend systems. Excalidraw visual tracking infrastructure shipped to `node_notes/diagrams/` (D-042). VACE_INPAINT_NODE_AUDIT.md gap analysis: 5 missing sections flagged. Phase 1 design principle locked: math-first before sampler integration (D-043).
- **2026-04-17:** Triple-repo cross-cutting day. (1) Slug alignment across gemini_video_course.py + nodes.py + AVM vlm_sam3_bridge.py (D-044). (2) AVM OpenRouter provider feature branch pushed. (3) SAM3VideoSegmentation widget bug solved structurally via Option A' — prompt_mode dropdown removed, input-presence auto-detection (D-047, supersedes D-045). (4) WAN VAE streaming helpers rewritten against current core API — encoder 2-frame chunks with `final=` flag, decoder list-of-chunks return (D-046). Unblocks VacePrePassReference and all other streaming-VAE callers.
- **2026-04-18:** NV_GeminiBatchExtractor production hardening. Three layered passes shipped + multi-AI reviewed in one session: (1) Phase 2 length safety + observability (`_check_extraction_health()`, schema_version cache invalidation, partial-on-failure caching to `_partial.json`, `finish_reason` normalized across Gemini direct + OR — D-048/048a). (2) Auto-token scaling (`_scale_max_tokens_by_duration()` formula 280 tok/min × 1.25, new `max_tokens_mode` dropdown, per-video `out_max=` in dry-run preview — D-049). (3) OR audit endpoints (`/api/v1/generation/{id}` post-call lookup with exponential backoff for settled cost + provider attribution + native tokens, `/api/v1/credits` warn-only pre-flight — D-050/051). Multi-AI review caught 7 critical bugs all fixed (settled_cost ≥ 0 vs > 0, defensive isinstance(dict) before mutation, soft-warn vs hard-raise on credit shortfall, stale partial cleanup, structured credits errors, lookup status surfaced, succeeded/failed cost split). Course pipeline workflow clarified (D-052): extract→SKILL is production, synth NODE is smoke test only. **Same-day runtime validation** exposed OR transcode GOP=1 bug — files didn't shrink (some grew 6-9%), 502 gateway timeouts on oversized payloads. Shipped same-session: aggressive new profile 1fps/GOP=60/720p/CRF30/32kbps mono audio (predicts all 36 corpus videos fit under 48MB raw), predictive size gate, pre-flight oversize hard-fail with 4-alternatives manifest, ComfyUI Cancel button responsiveness via `throw_exception_if_processing_interrupted()`, Windows subprocess UTF-8 fix (3 sites) — D-053/054/055.
- **2026-04-21:** NV_GeminiYoutubeBatchExtractor shipped (~400 LOC) — Gemini-direct YouTube batching with multiline URL input, `#` comments, dedup by canonical URL, dry-run preview with titles + cache status, schema-versioned cache + partial-on-failure + Cancel-responsive (D-056). Title-based JSON naming applied to both single + batch extractors via public YouTube oEmbed lookup (`https://www.youtube.com/oembed`, no auth, soft-fail) — `{sanitized_title}_{video_id}.json` preserving Unicode/emoji/CJK with 120-char cap and ID suffix for collision safety. Legacy `youtube_<id>.json` auto-migration runs pre-cache-check so prior extractions reuse (D-057). Runtime-exposed + same-session fixed the Gemini 3.x `thinkingLevel "disabled"` enum rejection on gemini-3.1-pro-preview (enum tightened — `"low"` is the new minimum for json_mode, D-058) + overly-aggressive 400 error-mapper that had been masking the real config error behind a misleading "likely private video" message (D-059). Stale `thinking=disabled` log string fixed to read payload dynamically.
- **2026-04-21 (late session):** Frontend Shift+Alt+Drag clone-with-connections fixed for ComfyUI frontend 1.42.10 (was broken on the 1.39→1.42 upgrade). Rewrote `web/clone_with_connections.js` to patch `LGraphCanvas.prototype._deserializeItems` — the chokepoint both Vue-nodes-mode and classic-canvas clone paths converge on — and leverage the native `connectInputs` flag + `LiteGraph.ctrl_shift_v_paste_connect_unselected_outputs` setting rather than manual `connect()` reconstruction. Single Ctrl+Z undo preserved. Establishes D-060 pattern for future frontend API patches.

## Archived Workstreams Index
<!-- Pointers to workstreams moved to ARCHIVE.md. -->

*No archived workstreams.*
