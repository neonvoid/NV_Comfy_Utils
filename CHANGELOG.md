# Changelog

All notable changes to NV_Comfy_Utils are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- NodeBypasser action widgets (Bypass/Enable/List) changed from BOOLEAN toggles to button widgets for reliable click handling. Existing nodes must be re-added.
- `extensions.js` now uses dynamic imports with per-module error isolation — one broken extension no longer kills all others.

### Fixed

- Fixed InpaintStitch2 pasting stabilized face at wrong position (double-head artifact) when CoTrackerBridge stabilization is used with InpaintCrop2's target_mode resize. Root cause: inverse content warp dx/dy were computed at target resolution but applied after resize to canvas scale, amplifying the shift by the resize ratio. Fix: apply inverse warp before resize.
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
