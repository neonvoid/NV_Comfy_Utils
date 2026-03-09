# Changelog

All notable changes to NV_Comfy_Utils are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- NV_VacePrePassReference: optional `identity_anchor` input for cross-chunk identity lock. Wire chunk 0's Kling output to anchor identity across all chunks. Prepended at t=0 (WAN 2.2 training prior), with sharpness quality floor. Original `reference_frames` input unchanged (current chunk's Kling output). IFS adaptive sampling available for both pools.
- Variables Pool system — each variable can have multiple candidate source connections. Switch between candidates via pool chips in the Variables Panel. Right-click any node output → "Add to Variable Pool" to register candidates.
- `healPool()` migration — automatically repairs orphaned pool entries and stale metadata from older variable system versions on workflow load.
- Pool purge controls — refresh button purges all stale candidates; per-variable "Purge Stale Pool Entries" in row context menu.
- Rebind UI for deleted source nodes — pool chips show dashed amber border when a candidate's source was deleted but a matching replacement node exists. Click to accept rebind or purge via context menu.

### Changed

- All variable/pool mutations now support Ctrl+Z undo via `_withUndo()` wrapper with reentrancy-safe nesting (LiteGraph `beforeChange()`/`afterChange()`).
- `getPool()` is now fully non-mutating — returns status annotations (`ok`/`stale`/`rebindable`/`type_mismatch`) and rebind suggestions without modifying candidate data. Mutations require explicit `rebindCandidate()` or `purgeStale()` calls.
- `_fuzzyRebind()` hardened — validates output type match and rejects ambiguous matches (>1 candidate with same nodeType + title + slotIndex). Previously could silently mis-bind to wrong node.
- Variables Panel refresh button now purges stale pool entries in addition to refreshing the UI.
- Pool hash includes alive/total candidate counts — panel auto-refreshes when source nodes are added or deleted.

### Fixed

- Variables Pool: deleted source nodes no longer silently rebind during read operations. Previously `getPool()` would mutate `candidate.nodeId` via `_fuzzyRebind()`, potentially binding to the wrong node without user confirmation.
- Variables Pool: innerHTML XSS vulnerability in drag-and-drop ghost element — now uses `createElement`/`textContent` for user-controlled variable names.
- Variables Pool: stale candidates no longer accumulate forever in workflow JSON — purge controls and `healPool()` migration clean up dead references.

### Added

- NV VACE Temporal Align node — automatically aligns all VACE conditioning entries to a common temporal dimension. Required when combining multiple VACE sources (e.g., WanVaceToVideo + NV_VacePrePassReference) that produce entries with different frame counts. Place between VACE conditioning chain and the sampler.
- Macro Groups in Quick Toggle floating panel — bundle multiple ComfyUI groups into a single toggle. Create/edit/delete macros via dialog with group checkbox list. Tri-state indicators (green=all on, amber=partial, red=all off) with `[enabled/total]` count badge. Collapsible macro sections with indented child rows. Macros persist in separate localStorage key (`NV_FloatingPanel_Macros`).
- MacroManager class separates macro CRUD/state logic from panel rendering. Versioned storage schema with migration on load.

### Changed

- Quick Toggle panel now has three sections when macros exist: MACROS → GROUPS (ungrouped) → CUSTOM PATTERNS. Groups assigned to macros appear only under their macro, not in the ungrouped list.
- All group/pattern toggle operations now wrapped in `beforeChange()`/`afterChange()` for undo batching (single Ctrl+Z undoes a macro toggle instead of N individual changes).
- Pattern dialog help text changed from innerHTML to textContent (XSS hardening).
- Polling loop now wrapped in try/catch (one refresh error no longer kills the loop permanently).
- Polling pauses while any dialog is open (prevents stale DOM references and checkbox state loss).
- `hide()` now cleans up any open dialogs via `_activeDialogs` tracking.
- Dialog z-index bumped: overlay=100002, dialog=100003 (was 100001/100002).
- Dirty-check hash added to `refreshGroups()` — skips DOM rebuild when group titles, modes, and macro state haven't changed.

### Fixed

- Variables system completely broken — `graph.links[id]` returned undefined because ComfyUI switched to `graph._links` (a Map). GetVariableNode data flow, NodeBypasser input reading, and LinkSwitcher all silently failed. Fixed all 7 usages across 4 JS files to use `graph._links.get()` with fallback.
- SetVariableNode visible on canvas at [-5000, Y] — `onDrawForeground`/`onDrawBackground` overrides only suppressed hook drawing, not the main node body rendered by `LGraphCanvas.drawNode()`. Added canvas patches to skip rendering, connection lines, and selection for managed setters.
- GetVariableNode `getInputLink` used `setter.inputs[slot]` where `slot` is the output slot index — now always uses `setter.inputs[0]` (the setter's single input) regardless of output slot queried.
- GetVariableNode output type always showed `*` (wildcard) instead of the actual data type — `updateType()` checked `setter.inputs[0].type` which stays `*` on wildcard slots. Now follows the link to the source node's actual output type (e.g., IMAGE, MODEL, LATENT).
- NV_MultiBandBlendStitch — standalone Laplacian pyramid multi-band blending node for stitch seam repair. Decomposes images into frequency bands and blends each at appropriate spatial scale. Pure PyTorch, ~2-5ms per frame.
- NV_BoundaryColorMatch — Reinhard color transfer at stitch boundaries in Lab color space. Samples color stats from strips on both sides of the seam and applies mean/std matching with gradient falloff. Temporal smoothing (default 0.8) prevents per-frame flicker on video.
- InpaintStitch2 now supports `blend_mode` parameter: `alpha` (default, backward compat), `multiband` (Laplacian pyramid — best for VAE roundtrip seams), and `hard` (binary paste). Multiband mode blends low frequencies broadly and high frequencies narrowly, directly at the composite step — no double-blending.
- NV_StitchBoundaryMask — per-frame gradient mask along stitch boundaries for boundary diffusion seam harmonization. Accepts bbox_mask (per-frame tracking), LATENT_STITCHER (static latent crop), or pixel STITCHER (per-frame canvas coordinates). Use with SetLatentNoiseMask + low-denoise KSampler to let WAN 2.2's native mask handling harmonize inpaint seams. Works for both latent-path (LatentInpaintStitch hard paste) and pixel-path (Kling API → InpaintStitch2) workflows.
- NV_KlingStitchAdapter — bridges Kling API output back to InpaintStitch2. Handles resolution mismatch (resizes to crop target), frame count mismatch (nearest-frame resampling), and validates against stitcher data. Wire between NV Kling Edit Video and NV Inpaint Stitch v2.
- NV_KlingUploadPreview now auto-fits arbitrary crop sizes to Kling-friendly dimensions (720-2160px, even, aspect-preserved). Stores original crop resolution in upload_config for downstream use.
- InpaintCrop2 stitcher now includes `crop_target_w/h` for downstream resolution bridging.

### Changed

- NodeBypasser action widgets (Bypass/Enable/List) changed from BOOLEAN toggles to button widgets for reliable click handling. Existing nodes must be re-added.
- `extensions.js` now uses dynamic imports with per-module error isolation — one broken extension no longer kills all others.

- InpaintStitch2 now outputs `stitch_mask` — per-frame mask [B,H,W] at full canvas resolution showing where inpainted content was composited (1.0=inpainted, 0.0=original). Feed directly into NV_BoundaryColorMatch or NV_StitchBoundaryMask.
- NV_VaceControlVideoPrep gains `halo_pixels` parameter (Seam-Absorbing Control Halo). Expands the VACE conditioning mask outward by N pixels beyond the stitch boundary so WAN repaints across the seam — eliminates seam memory in downstream stages. Default 0 (off, backward-compatible), recommended 8-16px.
- NV_BoundaryColorMatch now runs on GPU when available — was CPU-only, causing 20-40s processing time for 105-frame video batches. Now sub-second.

### Removed

- Deregistered NV_PixelToLatentStitcher — architecturally broken (SHELVED). File remains on disk but node no longer appears in ComfyUI.
- Removed unused `blend_power`/`iou_blend_power` parameter chain from NV_TemporalMaskStabilizer (was documented "kept for API compat" but never wired to any behavior).
- Removed dead first-attempt code in NV_StitchBoundaryMask `_gaussian_blur_batch`.
- Removed unused `MASK_CONFIG_TOOLTIP` constant from mask_processing_config.py.
- Deduplicated `compute_inscribed_radius` — vace_mask_prep.py now imports from vace_control_video_prep.py.

### Fixed

- Fixed NV_StitchBoundaryMask `pixel_stitcher` mode producing boundary mask around the entire frame edge instead of the stitch boundary — was using `canvas_to_orig` coordinates (original-in-canvas) instead of deriving crop-in-original coordinates from `cropped_to_canvas - canvas_to_orig`.
- Fixed NV_VacePrePassReference `ref_scale` blowing out latent values when `strength` is near-zero — `ref_strength / strength` now clamped to max 10x.
- Fixed `hard` blend mode in InpaintStitch2 being identical to `alpha` mode — now correctly thresholds the mask at 0.5 for true binary paste with no feathering.
- Fixed `mask_smooth` in InpaintCrop2 crashing on 2D mask input — `binary` was computed before `unsqueeze`, creating wrong shape for `gaussian_blur`. Now ensures 3D before binarization.
- Fixed VACE bbox stitch desync in VaceControlVideoPrep — stitch mask recomputed bboxes from raw input mask instead of preprocessed mask (after threshold/grow/fill_holes/smooth), causing mismatch with the VACE conditioning mask.
- Fixed CoTracker `grid_sample` using `padding_mode='zeros'` which injected dark pixels at warp edges. Changed to `padding_mode='border'` (edge replication).
- Fixed CoTracker error path not calling `model.cpu()`, leaking GPU memory on inference failure.
- Fixed snap_to_vae_grid in NV_LatentInpaintCrop clamping dimensions before origin — could theoretically produce out-of-bounds crop if called with unclamped coordinates. Now clamps origin first, then dimensions.
- Fixed FloatingPanel (Quick Toggle) becoming invisible when its saved position was off-screen — `show()` now auto-resets position to default if outside the viewport
- Fixed NV_PointPicker placing points at wrong coordinates when canvas aspect ratio doesn't match the available display area — mouse mapping now accounts for `object-fit: contain` letterboxing
- Fixed InpaintStitch2 pasting stabilized face at wrong position (double-head artifact) when CoTrackerBridge stabilization is used with InpaintCrop2's target_mode resize. Root cause: inverse content warp dx/dy were computed at target resolution but applied after resize to canvas scale, amplifying the shift by the resize ratio. Also fixed blend mask being incorrectly inverse-warped (causing halo/seam) — the blend mask from InpaintCrop is already in original coordinates and should not be warped. (see `bug_tracker/inpaint_stitch/2026-03-05_inverse_warp_resolution_mismatch.md`)
- Fixed NodeBypasser buttons silently failing on ComfyUI frontend v1.39+ — `onWidgetChange` was never called because the frontend uses `onWidgetChanged` (with 'd') and a different signature (see `bug_tracker/node_bypasser/2026-03-05_button_click_and_load_failures.md`)
- Fixed NodeBypasser intermittently disappearing after restart — class definition at module scope could race with `LGraphNode` global availability; now deferred to `registerCustomNodes()`

### Added

- NV_MaskProcessingConfig — shared config bus for mask processing parameters. Connect to InpaintCrop2, LatentInpaintCrop, MaskPipelineViz, or VaceControlVideoPrep to ensure identical mask settings across the pipeline. Fully backward compatible (optional input, nodes work as before when disconnected).
- NV_MaskPipelineViz: new `mode` selector (grid/batch/video). Batch mode outputs 6 separate images as a clickable batch. Video mode outputs all frames for a single selected stage for temporal scrub.
- NV_MaskPipelineViz: new `cropped_image` + `cropped_mask` optional inputs. Wire from InpaintCrop to preview mask stages on the actual crop the diffusion model sees.
- NV_PointPicker — interactive point placement node for CoTracker stabilization. Click on features in the cropped image to specify tracking points. Outputs JSON coordinates consumed by NV_CoTrackerBridge. Supports `frame_index` input to pick which video frame to annotate, plus `frame_index`/`total_frames` outputs for downstream chaining.
- NV_CoTrackerBridge now supports multi-point tracking via `tracking_points` input from NV_PointPicker. Averages trajectories across all tracked points weighted by visibility for more robust stabilization.
- NV_MaskTrackingBBox: new `ema` smoothing mode — bidirectional exponential moving average with single `alpha` parameter. Zero lag from forward+backward pass averaging.
- NV_MaskTrackingBBox: new `smooth_strength` parameter (0-1) — lerps between raw and smoothed coordinates. Works with all smoothing modes for easy partial-smoothing control.
- NV_AspectChunkPlanner — temporal segmentation by bounding-box aspect ratio changes. Analyzes per-frame bbox masks, detects aspect change points via log-ratio threshold, and segments the video into chunks where each has a stable aspect. Per-chunk union bbox and WAN-aligned target resolution exported as JSON plan file. Greedy segmentation with small-segment merge and optional WAN frame alignment.
- NV_TemporalMaskStabilizer — multi-stage temporal mask stabilization that fixes SAM3 mask pops. 5-stage pipeline: RAFT optical flow, flow-warped temporal consensus, SDF boundary smoothing, IoU outlier detection, and guided filter edge refinement. Optional bbox_mask input for cropped processing (faster, higher detail). Integrates with NV_MaskProcessingConfig for shared spatial cleanup settings. Binary/soft output modes.

## [0.1.0] - 2026-02-26

Initial tracked release. Captures all major capabilities built over the first 6 months of development.

### Added

- Chunked video pipeline v2 (Planner → ChunkLoader → KSampler → SaveChunkLatent → LatentStitcher → BoundaryNoiseMask → KSampler → StreamingVAEDecode)
- VRAM-aware chunk planner with auto-compute for max frames and overlap via binary search
- Boundary denoise pass for seamless chunk transitions (optimal range 0.10–0.20)
- Plan JSON v2.0 schema with `latent_stitch_config`, `vram_info`, per-chunk `latent_frames`
- Streaming VAE encode/decode nodes for memory-efficient video processing
- VACE pre-pass reference pipeline for video-to-video workflows
- Latent temporal utilities (PrependReferenceLatent, TemporalConcat, TemporalSlice)
- Parameter sweep system (SweepPlanner, SweepLoader, SweepRecorder)
- WAN VRAM estimator for inference peak memory prediction
- Context window optimizer and sampler patch for attention management
- Variables system (frontend JS) for dynamic workflow parameters
- Node bypasser for selective node enable/disable during execution
- Stable naming system for persistent node identification
- Interactive bounding box creator with frontend UI
- Slack notification and error handling integration
- B2 cloud input/output sync nodes
- Prompt library and prompt refiner nodes
- Pipeline benchmark and generation timer nodes
- Preview animation player with download support
- JSON metadata reader/writer nodes
- Frame number and text overlay nodes

### Fixed

- Noise bleed (magenta/green color cast) in first 1–2 frames of V2V renders caused by `ref_latent` being all zeros in VacePrePassReference
- Temporal nodes (PrependReferenceLatent, TemporalConcat, TemporalSlice) now build clean output dicts instead of shallow-copying stale metadata keys (`noise_mask`, `batch_index`)
- Inpaint crop/repaint coordinate bugs
- Streaming VAE encode/decode stability fixes
- Context window monkeypatch fixes
- VACE slicer context fix to accept reference images
- Committed noise handling for VACE long-form tests

### Deprecated

- `co_denoise_sampler.py` — hardcodes Euler scheduler; use standard KSampler pipeline instead
- `chunk_stitcher.py` — pixel-space stitching causes ghosting; use LatentStitcher instead
