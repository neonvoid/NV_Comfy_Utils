# Changelog

All notable changes to NV_Comfy_Utils are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
