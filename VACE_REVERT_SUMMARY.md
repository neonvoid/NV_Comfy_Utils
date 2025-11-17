# VACE Channel Fix Revert - November 2024

## Issue

An incorrect "fix" was applied that reduced VACE control channels from 32 to 16, breaking the model's expected input format.

**Error caused by incorrect fix:**
```
RuntimeError: Given groups=1, weight of size [5120, 96, 1, 2, 2],
expected input[1, 80, 22, 104, 60] to have 96 channels, but got 80 channels instead
```

## Root Cause

The WAN model's `vace_patch_embedding` Conv3d layer is trained to expect **96 input channels**:
- **32 control channels**: inactive (16) + reactive (16) concatenated
- **64 mask channels**: 8×8 spatial mask (vae_stride² = 8² = 64)
- **Total: 96 channels**

The incorrect fix changed control from 32 to 16 channels, resulting in only 80 total channels (16 + 64).

## What Was Reverted

### 1. Main VACE Encoding (nodes.py, line ~3458)

**Incorrect Code:**
```python
chunk_vace_frames = reactive_batched  # [1, 16, T_lat, H, W] - WRONG!
```

**Corrected Code:**
```python
chunk_vace_frames = torch.cat([inactive_batched, reactive_batched], dim=1)  # [1, 32, T_lat, H, W]
```

### 2. Tier 3 VACE Reference (nodes.py, line ~4234)

**Incorrect Code:**
```python
vace_reference = last_frame_latent  # [1, 16, 1, H, W] - WRONG!
```

**Corrected Code:**
```python
zeros_padding = torch.zeros_like(last_frame_latent)
vace_reference = torch.cat([last_frame_latent, zeros_padding], dim=1)  # [1, 32, 1, H, W]
```

### 3. Start Image VACE Reference (nodes.py, line ~6149)

**Incorrect Code:**
```python
vace_reference_latent = reference_encoded  # [1, 16, 1, H, W] - WRONG!
```

**Corrected Code:**
```python
zeros_padding = torch.zeros_like(reference_encoded)
vace_reference_latent = torch.cat([reference_encoded, zeros_padding], dim=1)  # [1, 32, 1, H, W]
```

## Why 32 Channels Are Required

The WAN VACE architecture uses a **dual-stream approach**:

1. **Inactive Stream (16 channels)**
   - Preserves areas not under active control
   - Computed as: `(frames - 0.5) * (1 - mask) + 0.5`
   - Encodes to 16 latent channels via VAE

2. **Reactive Stream (16 channels)**
   - Controls areas being actively modified
   - Computed as: `(frames - 0.5) * mask + 0.5`
   - Encodes to 16 latent channels via VAE

3. **Concatenation (32 channels)**
   - Combined: `torch.cat([inactive, reactive], dim=1)`
   - This is the trained model architecture
   - Cannot be changed without retraining the model

4. **Mask Channels (64 channels)**
   - Added separately: 8×8 spatial mask per latent position
   - Total pipeline: 32 (control) + 64 (mask) = **96 channels**

## What Was NOT Changed

All the important fixes from BUGFIX_SUMMARY_2025.md remain intact:

✅ **Dynamic VAE Frame Counting**
- No hardcoded 16-frame assumptions
- Extracts last N latent frames first
- Decodes to get actual video frame count
- Tracks actual overlap with `self._actual_overlap_frames`

✅ **VACE Reference Frame Handling**
- Correctly skips reference frame when `has_vace_reference=True`
- Extracts VACE controls from `chunk_positive[0][1]["vace_frames"]`
- Removes first frame if reference was prepended

✅ **Actual Overlap Tracking**
- Stores `self._actual_overlap_frames` from VAE output
- Stores `self._actual_vace_overlap_frames` for VACE controls
- Uses actual values instead of hardcoded assumptions

✅ **Dynamic Frame Count Calculation**
- Iterates through actual chunk sizes
- Calculates total frames from real data
- No formula-based assumptions

## Architectural Understanding

The dual-stream (inactive + reactive) architecture is fundamental to how WAN VACE works:

```
Control Video (RGB) + Mask
         ↓
Split into two streams:
  Inactive: (video - 0.5) * (1 - mask) + 0.5  → VAE encode → 16 channels
  Reactive: (video - 0.5) * mask + 0.5        → VAE encode → 16 channels
         ↓
Concatenate: [inactive, reactive] → 32 channels
         ↓
Add spatial mask: 64 channels (8×8 per position)
         ↓
vace_patch_embedding expects: 96 channels total
```

This cannot be simplified to a single stream without retraining the model.

## Stencil Implementation

The stencil support (path-based inputs) is **separate and unaffected** by this revert:
- ✅ Stencil path inputs remain functional
- ✅ JSON structure with stencil paths preserved
- ✅ Triple-channel encoding method `_encode_vace_triple_channel()` still available for future use

## Verification

After this revert, you should:

1. ✅ No longer see "96 vs 80 channels" error
2. ✅ VACE controls work normally
3. ✅ All temporal consistency tiers function correctly
4. ✅ Dynamic frame counting still works (no hardcoded values)

## Lessons Learned

1. **Never change channel counts** without understanding the trained model architecture
2. **96-channel input is required**: 32 (control) + 64 (mask)
3. **Dual-stream is fundamental**: inactive + reactive must be concatenated
4. **Spatial dimension errors** are different from channel count errors
5. **Always verify against model weights**: `vace_patch_embedding.weight.shape[1]` tells you expected input channels

## Status

✅ **Revert Complete**
✅ **All Original Fixes Preserved**
✅ **Model Architecture Restored**
✅ **Ready for Testing**

---

*Date: November 2024*
*Files Modified: nodes.py (3 locations)*
*Status: Reverted incorrect fix, restored correct 32-channel architecture*