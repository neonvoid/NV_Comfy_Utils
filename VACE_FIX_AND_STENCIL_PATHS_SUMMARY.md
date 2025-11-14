# VACE Dimension Fix & Stencil Path Implementation

## Date: November 2024

---

## Issues Fixed

### 1. VACE Dimension Mismatch Error ✅

**Error:**
```
RuntimeError: The size of tensor a (32760) must match the size of tensor b (9360) at non-singleton dimension 1
```

**Root Cause:**
The dual-channel VACE encoding was concatenating inactive + reactive latents to create 32 channels, but the WAN model expects only 16 channels for control input.

**Fix Applied:**
Changed from concatenation to using only the reactive stream (16 channels) in 3 locations:
1. Line 3459: Main VACE encoding - use `reactive_batched` only
2. Line 4234: Tier 3 VACE reference - use `last_frame_latent` only
3. Line 6009: Start image VACE reference - use `reference_encoded` only

---

## New Features

### 2. Stencil Path-Based Input Support ✅

Implemented stencil support using file paths (consistent with other control videos like depth, openpose, etc.).

**Design Decision:** Path-based inputs instead of direct IMAGE/MASK nodes
- ✅ Consistent with existing control video inputs
- ✅ No memory overhead from storing tensors in memory
- ✅ Easier file-based workflow management
- ✅ Simpler JSON structure (just paths, no binary data)

#### A. Input Parameters (NV_VideoChunkAnalyzer)

```python
"stencil_video_path": ("STRING", {
    "default": "",
    "tooltip": "Optional: Path to stencil video for preservation areas (e.g. faces, text)"
}),
"stencil_mask_path": ("STRING", {
    "default": "",
    "tooltip": "Optional: Path to stencil mask video (1=preserve, 0=generate)"
}),
```

#### B. Validation Logic

```python
if stencil_video_path.strip() or stencil_mask_path.strip():
    if not (stencil_video_path.strip() and stencil_mask_path.strip()):
        raise ValueError("Both paths must be provided together")

    if not os.path.exists(stencil_video_path):
        raise ValueError(f"Stencil video not found: {stencil_video_path}")
    if not os.path.exists(stencil_mask_path):
        raise ValueError(f"Stencil mask not found: {stencil_mask_path}")
```

#### C. JSON Structure

```json
{
  "stencil": {
    "has_stencil": true,
    "video_path": "D:/path/to/stencil.mp4",
    "mask_path": "D:/path/to/mask.mp4",
    "mode": "preserve",
    "description": "Stencil areas (mask=1) will be preserved during generation"
  },
  "chunks": [
    {
      "stencil": {
        "enabled": true,
        "mode": "preserve",
        "weight": 1.0
      }
    }
  ]
}
```

#### D. Triple-Channel Encoding Method

The `_encode_vace_triple_channel()` method (lines 4930-5017) blends three streams:

1. **Inactive**: Neutral gray (no control)
2. **Reactive**: Active control areas not covered by stencil
3. **Stencil**: High-priority preservation areas

**Blending Formula:**
```python
blended = (inactive * inactive_weight +
           reactive * reactive_weight +
           stencil * stencil_weight)
```

---

## Files Modified

### nodes.py - All Changes

1. **Lines 5415-5422**: Changed stencil inputs to STRING paths
   - From IMAGE/MASK to STRING (path) inputs

2. **Lines 5431-5434**: Updated analyze() method signature
   - Changed parameters from `stencil_video`, `stencil_mask` to `stencil_video_path`, `stencil_mask_path`

3. **Lines 5471-5488**: Updated validation logic
   - Check for non-empty path strings
   - Validate paths exist on filesystem
   - Store paths in stencil_info dict

4. **Lines 5514-5516**: Updated info output
   - Display stencil video and mask filenames

5. **Lines 5552-5557**: Chunk stencil configuration
   - Per-chunk stencil settings (enabled, mode, weight)

6. **Lines 5567-5571**: Removed tensor-based chunk saving
   - No longer saves stencil chunks (uses original files via paths)

7. **Lines 5602-5610**: Updated JSON stencil structure
   - Includes video_path and mask_path
   - Global stencil configuration

8. **Lines 3452-3459**: Fixed VACE channel concatenation
   - Main encoding uses 16 channels only

9. **Lines 4232-4234**: Fixed Tier 3 VACE reference
   - Uses 16 channels (removed zero padding)

10. **Lines 4930-5017**: Added `_encode_vace_triple_channel()` method
    - Complete stencil-aware encoding implementation

---

## Usage Example

### Step 1: Prepare Stencil Files

Create two video files:
- **Stencil Video**: Contains the content to preserve (e.g., face regions)
- **Stencil Mask**: Binary mask where white (1.0) = preserve, black (0.0) = allow generation

### Step 2: Configure NV_VideoChunkAnalyzer

```python
# In ComfyUI workflow:
NV_VideoChunkAnalyzer:
  - images: [your main video frames]
  - stencil_video_path: "D:/project/stencil_faces.mp4"
  - stencil_mask_path: "D:/project/mask_faces.mp4"
  - ... (other parameters)
```

### Step 3: Generated JSON

```json
{
  "metadata": { ... },
  "control_videos": {
    "openpose": "D:/controls/openpose.mp4",
    "depth": "D:/controls/depth.mp4"
  },
  "stencil": {
    "has_stencil": true,
    "video_path": "D:/project/stencil_faces.mp4",
    "mask_path": "D:/project/mask_faces.mp4",
    "mode": "preserve"
  },
  "chunks": [ ... ]
}
```

### Step 4: Preprocessing

The `NV_ChunkConditioningPreprocessor` will:
1. Load stencil video and mask from paths
2. Use `_encode_vace_triple_channel()` to blend three streams
3. Inject blended 16-channel control into conditioning

---

## Benefits

### Consistency
- All video inputs (beauty, depth, openpose, stencil) use the same path-based approach
- Unified workflow for managing control videos

### Efficiency
- No memory overhead from storing large video tensors
- Reference original files instead of duplicating data
- Smaller JSON files (paths vs binary data)

### Flexibility
- Easy to swap stencil files without reprocessing
- Per-chunk enable/disable stencil control
- Adjustable stencil weight (0.0 = disabled, 1.0 = full preservation)

### Triple-Channel Blending
- **Priority system**: Stencil > Reactive > Inactive
- **Smooth transitions**: Weighted blending prevents hard edges
- **Fine-grained control**: Preserve specific regions while allowing generation elsewhere

---

## Testing Recommendations

### 1. Test VACE Dimension Fix
```bash
# Run standard video generation workflow
# Should NOT see dimension mismatch errors
# Console should show: "shape=torch.Size([1, 16, ...])" (not 32)
```

### 2. Test Stencil Without Files
```bash
# Leave stencil paths empty - should work normally
# JSON should not contain stencil section
```

### 3. Test Stencil With Files
```bash
# Provide both stencil_video_path and stencil_mask_path
# JSON should contain stencil section with paths
# Verify paths are valid and files exist
```

### 4. Test Error Handling
```bash
# Try providing only one path - should error
# Try non-existent paths - should error
# Try mismatched video/mask dimensions - should error (in preprocessor)
```

---

## Console Output Examples

### Fixed VACE Output
```
[VACE Dual] Inactive latents: torch.Size([16, 21, 104, 60])
[VACE Dual] Reactive latents: torch.Size([16, 21, 104, 60])
✓ VACE reference encoded: torch.Size([1, 16, 1, 104, 60])
    Structure: 16ch VAE latent
```

### Stencil Path Configuration
```
============================================================
NV VIDEO CHUNK ANALYZER
============================================================
Input: 243 frames
Resolution: 832×480
Chunk size: 81 frames
Chunk overlap: 16 frames
Stride: 65 frames
Control videos: 2 detected
  - openpose: openpose.mp4
  - depth: depth.mp4
Stencil: stencil_faces.mp4 (mode: preserve)
  Mask: mask_faces.mp4
```

### Triple-Channel Encoding (Future)
```
[VACE Triple] Encoding with stencil support
[VACE Triple] Input frames shape: torch.Size([81, 832, 480, 3])
[VACE Triple] Stencil video shape: torch.Size([81, 832, 480, 3])
[VACE Triple] Blend weights: stencil=0.250, reactive=0.500, inactive=0.250
[VACE Triple] ✓ Encoding complete with stencil blending
```

---

## Status

✅ **VACE Dimension Fix**: Complete
✅ **Stencil Path Inputs**: Complete
✅ **JSON Structure**: Complete
✅ **Triple-Channel Encoding Method**: Complete (ready for preprocessor integration)
⏳ **Preprocessor Integration**: Pending (load stencil from paths)
⏳ **User Testing**: Ready for testing

---

## Next Steps

1. **Update NV_ChunkConditioningPreprocessor** to:
   - Load stencil video/mask from JSON paths
   - Call `_encode_vace_triple_channel()` when stencil is present
   - Fall back to `_encode_vace_dual_channel()` when no stencil

2. **Test Integration**:
   - Run full workflow with stencil files
   - Verify preservation areas work correctly
   - Compare quality with/without stencil

3. **Documentation**:
   - Add example workflows
   - Create stencil preparation guide
   - Document best practices for mask creation

---

## Key Takeaways

1. ✅ **Path-based inputs** provide consistency and efficiency
2. ✅ **16-channel VACE** fixes dimension mismatch errors
3. ✅ **Triple-channel blending** enables fine-grained preservation control
4. ✅ **JSON stores paths** not tensor data (simpler, more portable)
5. ✅ **Per-chunk control** allows flexible stencil usage

---

*Implementation: Claude Assistant*
*Date: November 2024*
*Status: Ready for Testing*