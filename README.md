# NV_Comfy_Utils

A collection of ComfyUI custom nodes including NodeBypasser and Custom Video Saver.

## 🎯 Features

### NodeBypasser
- **List All Nodes**: View all nodes in your current ComfyUI workflow
- **Text-Based Selection**: Type node names separated by commas to select which nodes to bypass/enable
- **Smart Matching**: Case-insensitive matching that works with both node types and titles
- **Bulk Operations**: Bypass or enable multiple nodes at once
- **Real-time Feedback**: See exactly which nodes were processed and which weren't found

### Custom Video Saver
- **Custom Directory Selection**: Save videos to any directory you choose
- **Multiple Video Formats**: Support for MP4, AVI, MOV, MKV, WebM, WMV
- **Quality Control**: Adjustable video quality settings
- **Color Preservation**: High-precision color data preservation with validation
- **Flexible Naming**: Custom filename prefixes with automatic counter
- **ComfyUI Integration**: Follows ComfyUI's native file saving patterns

## 🚀 How to Use

### NodeBypasser

#### 1. Add the Node
- Search for "NodeBypasser" in ComfyUI's node search
- Add it to your workflow

#### 2. List Available Nodes
- Click the **"List All Nodes"** button
- View all nodes in the Status area with their IDs, types, and current state

#### 3. Select Nodes to Bypass/Enable
- In the **"Node Names to Bypass"** text field, type the names of nodes you want to affect
- Separate multiple names with commas
- Examples:
  ```
  LoadImage, LoadVideo
  test1, test2
  KNF_Organizer
  ```

#### 4. Execute Actions
- Click **"Bypass Nodes"** to bypass all matching nodes
- Click **"Enable Nodes"** to enable all matching nodes
- Check the Status area for results

### Custom Video Saver

#### 1. Add the Node
- Search for "Custom Video Saver" in ComfyUI's node search
- Add it to your workflow

#### 2. Connect Video Input
- Connect a video tensor (IMAGE type) to the `video_tensor` input
- This can come from any node that outputs video frames

#### 3. Configure Settings
- **Filename Prefix**: Set the base name for your video file (e.g., "my_video")
- **Custom Directory**: Specify where to save the video
  - Leave empty to use ComfyUI's default output directory
  - Use absolute path (e.g., `C:\Videos\MyProject`) for specific location
  - Use relative path (e.g., `videos\output`) relative to ComfyUI root
- **Video Format**: Choose from MP4, AVI, MOV, MKV, WebM, WMV
- **FPS**: Set frames per second (1.0 to 120.0)
- **Quality**: Set video quality (0=lossless, 18=high, 23=medium, 51=lowest)
- **Preserve Colors**: Enable high-precision color preservation (recommended: True)

#### 4. Optional Settings
- **Subfolder**: Add a subfolder within the output directory
- **Prompt**: Attach prompt information to the video metadata
- **Extra PNG Info**: Add additional metadata

#### 5. Execute
- Run your workflow
- The video will be saved to your specified location with automatic filename incrementing
- Files are numbered sequentially: `video_00001.mp4`, `video_00002.mp4`, etc.
- Check the output for the full file path and filename

### ✨ **Automatic Filename Incrementing**
The Custom Video Saver automatically increments filenames to prevent overwriting:
- **Sequential Numbering**: Files are saved as `prefix_00001.ext`, `prefix_00002.ext`, etc.
- **Gap Handling**: If files are deleted, the next available number is used
- **No Overwriting**: Your videos are always safe from accidental overwrites
- **Up to 99,999 Files**: Supports large batches with 5-digit counters

## 📝 Usage Examples

### NodeBypasser Examples

Based on your workflow, you can use:

| What to Type | Matches | Result |
|--------------|---------|--------|
| `test1` | Node 50 (LoadVideo - test1) | Bypasses the first LoadVideo node |
| `test2` | Node 51 (LoadVideo - test2) | Bypasses the second LoadVideo node |
| `LoadVideo` | Node 50 & 51 (both LoadVideo nodes) | Bypasses both LoadVideo nodes |
| `LoadImage` | Node 44 (LoadImage) | Bypasses the LoadImage node |
| `KNF_Organizer` | Node 47 (KNF_Organizer) | Bypasses the KNF_Organizer node |

### Custom Video Saver Examples

#### Directory Examples
```python
# Use default ComfyUI output directory
custom_directory = ""

# Use absolute path
custom_directory = "C:\\Videos\\MyProject"

# Use relative path from ComfyUI root
custom_directory = "videos\\output"

# Use subfolder
custom_directory = "C:\\Videos\\MyProject"
subfolder = "batch_001"
```

#### Video Format Examples
| Format | Use Case | Quality | File Size |
|--------|----------|---------|-----------|
| MP4 | General purpose, web compatible | High | Medium |
| AVI | Windows compatibility | High | Large |
| MOV | Mac compatibility | High | Large |
| MKV | High quality, metadata support | Very High | Large |
| WebM | Web streaming | Medium | Small |
| WMV | Windows Media | Medium | Small |

#### Quality Settings
| Quality Value | Description | Use Case |
|---------------|-------------|----------|
| 0 | Lossless | Archival, maximum quality |
| 18 | High | Professional work |
| 23 | Medium | General use, good balance |
| 28 | Low | Quick previews |
| 51 | Lowest | Minimal file size |

#### Color Preservation Settings
| Setting | Description | Use Case |
|---------|-------------|----------|
| **Preserve Colors: True** | High-precision color conversion with validation | Professional work, color-critical applications |
| **Preserve Colors: False** | Standard conversion (faster) | Quick previews, non-critical color work |

#### Color Data Handling
- **Float32/64 Input**: Uses high-precision rounding for accurate conversion
- **16-bit Input**: Properly scales down to 8-bit while preserving color information
- **Color Space Validation**: Validates color values before and after conversion
- **Channel Support**: Handles RGB, RGBA, and grayscale inputs correctly

#### Codec Requirements
The node automatically detects and uses the best available codec:

| Format | Primary Codec | Fallback Codecs |
|--------|---------------|-----------------|
| MP4 | MP4V | XVID, Motion JPEG |
| AVI | XVID | Motion JPEG, MP4V |
| MOV | MP4V | XVID, Motion JPEG |
| MKV | MP4V | XVID, Motion JPEG |
| WebM | VP8 | MP4V |
| WMV | WMV2 | MP4V |

**Note**: H.264 codec requires additional libraries and may not be available on all systems. The node will automatically fall back to more compatible codecs.

## 🔍 Smart Matching Rules

The system uses **case-insensitive partial matching** on both:
- **Node Type**: The internal ComfyUI node type (e.g., "LoadImage", "LoadVideo")
- **Node Title**: The display name shown in the workflow (e.g., "test1", "test2")

### Examples:
- `load` matches both "LoadImage" and "LoadVideo"
- `test` matches both "test1" and "test2"
- `image` matches "LoadImage"
- `organizer` matches "KNF_Organizer"

## 🏗️ Technical Architecture

### Custom Video Saver (Python Backend)
The Custom Video Saver is implemented as a **server-side** ComfyUI node:

```python
# File: src/KNF_Utils/nodes.py
class CustomVideoSaver:
    def save_video(self, video_tensor, filename_prefix, custom_directory, 
                   video_format, fps, quality, subfolder, prompt, extra_pnginfo):
        # Determine output directory (custom or default)
        if custom_directory and custom_directory.strip():
            if os.path.isabs(custom_directory):
                output_dir = custom_directory
            else:
                comfy_root = os.path.dirname(folder_paths.get_output_directory())
                output_dir = os.path.join(comfy_root, custom_directory)
        else:
            output_dir = folder_paths.get_output_directory()
        
        # Generate filename with counter (ComfyUI pattern)
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(filename_prefix, output_dir, 
                                           video_tensor.shape[2], video_tensor.shape[1])
        
        # Convert tensor to video file using OpenCV
        video_filename = f"{filename}_{counter:05}.{video_format}"
        video_path = os.path.join(full_output_folder, video_filename)
        
        return self._tensor_to_video_file(video_tensor, video_path, fps, quality, video_format)
```

### Frontend (JavaScript)
The NodeBypasser is implemented as a **frontend-only** ComfyUI extension:

```javascript
// File: web/node_bypasser.js
class NodeBypasser extends LGraphNode {
    constructor() {
        // Create widgets for user interaction
        this.listNodesButton = ComfyWidgets["BOOLEAN"](...);
        this.nodeNamesInput = ComfyWidgets["STRING"](...);
        this.bypassButton = ComfyWidgets["BOOLEAN"](...);
        this.enableButton = ComfyWidgets["BOOLEAN"](...);
    }
    
    // Access the live ComfyUI graph
    listAllNodes() {
        const graph = app.graph;
        const nodes = graph._nodes;
        // Process and display nodes
    }
    
    // Bypass nodes by name matching
    bypassNodesByName(bypass) {
        const nodeNames = this.nodeNamesInput.value.split(',');
        // Find matching nodes and set their mode
        targetNode.mode = bypass ? MODE_BYPASS : MODE_ALWAYS;
    }
}
```

### ComfyUI Integration
The node integrates with ComfyUI's frontend system:

```javascript
// Registration with ComfyUI
app.registerExtension({
    name: "NV_Comfy_Utils.NodeBypasser",
    registerCustomNodes() {
        LiteGraph.registerNodeType("NodeBypasser", NodeBypasser);
    }
});
```

### Node Modes
ComfyUI uses different node modes to control execution:
- **Mode 0 (ALWAYS)**: Node executes normally
- **Mode 4 (BYPASS)**: Node is bypassed (skipped during execution)
- **Mode 2 (MUTED)**: Node is muted (different from bypass)

## 🔧 How JavaScript Interacts with ComfyUI

### 1. **Graph Access**
```javascript
// Access the live workflow graph
const graph = app.graph;
const nodes = graph._nodes;
```

### 2. **Node Manipulation**
```javascript
// Change node execution mode
node.mode = MODE_BYPASS;  // Bypass the node
node.mode = MODE_ALWAYS;  // Enable the node
```

### 3. **Widget System**
```javascript
// Create interactive widgets
const widget = ComfyWidgets["STRING"](this, "name", ["STRING", options], app).widget;

// Handle widget changes
this.onWidgetChange = function(widget, value) {
    // React to user input
};
```

### 4. **Event Handling**
```javascript
// Override button click behavior
widget.onClick = (options) => {
    // Custom action when button is clicked
};
```

## 📁 File Structure

```
NV_Comfy_Utils/
├── __init__.py                 # Main entry point with WEB_DIRECTORY
├── web/
│   ├── extensions.js           # Frontend entry point
│   └── node_bypasser.js       # NodeBypasser implementation
├── src/KNF_Utils/
│   ├── __init__.py            # Python node exports
│   ├── nodes.py               # Python nodes (KNF_Organizer, CustomVideoSaver, etc.)
│   └── smart_video_loader.py  # Video path loader utilities
└── tests/
    └── test_KNF_Utils.py      # Unit tests for all nodes
```

## 🎨 Widget Types Used

| Widget Type | Purpose | Example |
|-------------|---------|---------|
| `BOOLEAN` | Toggle buttons | "List All Nodes", "Bypass Nodes" |
| `STRING` | Text input | "Node Names to Bypass", "Status", "Custom Directory" |
| `COMBO` | Dropdown selection | "Video Format" selection |
| `FLOAT` | Decimal numbers | "FPS" setting |
| `INT` | Integer numbers | "Quality" setting |

## 🚨 Troubleshooting

### NodeBypasser Issues

#### Node Not Appearing
- Ensure `WEB_DIRECTORY = "./web"` is set in `__init__.py`
- Restart ComfyUI completely
- Check browser console for errors

#### Widgets Not Showing
- Add a new NodeBypasser node (don't reuse old ones)
- Check console for widget creation messages
- Verify all widgets are created successfully

#### Bypassing Not Working
- Check that node names are typed correctly
- Use "List All Nodes" to see exact node names
- Check the Status area for error messages

### Custom Video Saver Issues

#### Filename Incrementing
- **Files being overwritten**: This has been fixed! The Custom Video Saver now automatically increments filenames (e.g., `video_00001.mp4`, `video_00002.mp4`, etc.)
- **Gap handling**: If files are deleted, the saver will find the next available number (e.g., if `00001.mp4` and `00003.mp4` exist, it will use `00002.mp4`)
- **Maximum files**: Supports up to 99,999 files per prefix (00001-99999)

#### Video Not Saving
- Check that the input is a valid video tensor (IMAGE type)
- Verify the custom directory path is correct and writable
- Check ComfyUI console for error messages
- Ensure OpenCV is properly installed with video codec support

#### Codec Issues
- **H.264 not available**: This is normal - the node will automatically fall back to MP4V
- **OpenH264 library errors**: Ignore these warnings - the video will still save successfully
- **"Failed to initialize VideoWriter"**: Try a different video format (AVI, MOV) or check OpenCV installation
- **No codecs available**: Reinstall OpenCV with video support: `pip install opencv-python`

#### Poor Video Quality
- Adjust the quality setting (lower numbers = higher quality)
- Try different video formats (MP4 usually works best)
- Check that FPS setting matches your source video

#### Directory Access Issues
- Use absolute paths for custom directories
- Ensure the directory exists or can be created
- Check file permissions for the target directory
- Try using ComfyUI's default output directory first

#### File Not Found After Saving
- Check the output path in the node's info output
- Verify the file was actually created
- Look for error messages in ComfyUI console

## 🔄 Development Notes

### Frontend-Only Design
Unlike traditional ComfyUI nodes that have both Python backend and JavaScript frontend, NodeBypasser is **frontend-only**:
- No Python execution needed
- Direct access to live graph state
- Immediate UI feedback
- No workflow execution required

### LiteGraph Integration
The node extends `LGraphNode` and integrates with LiteGraph's widget system:
- Automatic widget positioning
- Built-in event handling
- ComfyUI styling and behavior

This approach provides a more responsive user experience compared to traditional Python-based nodes that require workflow execution to function.