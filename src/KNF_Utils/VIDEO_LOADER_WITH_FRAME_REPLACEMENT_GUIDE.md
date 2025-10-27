# Video Loader with Frame Replacement

This custom node extends VideoHelperSuite's LoadVideoFFmpegPath with the ability to replace the first and/or last frame of a video with custom images.

## Features

- **Full VideoHelperSuite Compatibility**: Inherits all features from VHS's LoadVideoFFmpegPath including:
  - FFmpeg-based video loading (high quality, wide codec support)
  - Custom resolution and frame rate control
  - Frame loading cap and start time
  - Format presets (AnimateDiff, Mochi, LTXV, Hunyuan, etc.)
  - VAE encoding support
  - Audio extraction
  - Alpha channel/mask support

- **Frame Replacement**:
  - Replace first frame with a custom image
  - Replace last frame with a custom image
  - Automatic resizing to match video dimensions
  - Channel conversion (RGB/RGBA handling)

## Usage

### Basic Video Loading

Use the node just like VideoHelperSuite's LoadVideoFFmpegPath:

```
Required Inputs:
- video: Path to video file
- force_rate: Frame rate (0 = use original)
- custom_width: Custom width (0 = use original)
- custom_height: Custom height (0 = use original)  
- frame_load_cap: Max frames to load (0 = load all)
- start_time: Start time in seconds
```

### Frame Replacement

To replace frames, connect image tensors and enable the boolean switches:

```
Optional Inputs:
- first_frame_image: IMAGE tensor to use as first frame
- last_frame_image: IMAGE tensor to use as last frame
- replace_first_frame: BOOLEAN - Enable first frame replacement
- replace_last_frame: BOOLEAN - Enable last frame replacement
```

## Example Workflows

### Replace First Frame Only

1. Add "Video Loader with Frame Replacement" node
2. Set video path
3. Connect an image to `first_frame_image`
4. Enable `replace_first_frame` (set to True)
5. Keep `replace_last_frame` disabled (False)

### Replace Both First and Last Frames

1. Add "Video Loader with Frame Replacement" node
2. Set video path
3. Connect an image to `first_frame_image`
4. Connect another image to `last_frame_image`
5. Enable both `replace_first_frame` and `replace_last_frame`

### Use Case: Consistent Start/End Frames for Video Generation

When working with AI video generation models, you might want to:
- Use a specific first frame as a reference
- Ensure the video ends on a specific frame
- Create seamless loops by making first and last frames identical

## Technical Details

### Image Processing

The node automatically handles:
- **Resolution matching**: Images are resized to match video dimensions using bilinear interpolation
- **Channel conversion**: 
  - RGBA → RGB (drops alpha)
  - Grayscale → RGB (repeats channel)
  - RGB → RGBA (adds alpha channel)
- **Batch handling**: Takes first image from batch if multiple provided

### Compatibility

- **Requires**: comfyui-videohelpersuite must be installed
- **VAE Mode**: Frame replacement is NOT supported when using VAE encoding (latent output)
- **Mask Support**: Alpha channels are preserved and output as masks

## Outputs

Same as VideoHelperSuite's LoadVideoFFmpegPath:

1. **IMAGE/LATENT**: Video frames as image tensor (or latent if VAE provided)
2. **mask**: Alpha mask if video has alpha channel
3. **audio**: Extracted audio from video
4. **video_info**: Metadata (fps, resolution, frame count, duration)

## Notes

- Frame replacement only works with image output (not latent/VAE mode)
- Images are automatically resized to match video resolution
- Original video file is not modified
- The node uses VideoHelperSuite's robust FFmpeg backend for video loading

## Error Handling

- If VideoHelperSuite is not installed, the node will display an error message
- If frame replacement fails, a warning is printed but video loading continues
- Invalid images are caught and reported without crashing the workflow

## Category

`KNF_Utils/Video`

## Dependencies

- comfyui-videohelpersuite
- PyTorch
- NumPy

