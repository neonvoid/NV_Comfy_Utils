# Troubleshooting NV Video Loader Path

## Issue: Node appears in search but won't add to canvas

This happens when the node fails to import properly. Here's how to diagnose and fix it:

## Step 1: Check the Console for Error Messages

When you start ComfyUI, watch the console/terminal for these messages:

### ✅ **Success Messages** (what you should see):
```
[NV_Video_Loader_Path] Successfully imported VideoHelperSuite
[NV_Comfy_Utils] NV_Video_Loader_Path registered successfully
```

### ❌ **Error Messages** (what indicates a problem):
```
[NV_Video_Loader_Path] ERROR: Could not import VideoHelperSuite!
[NV_Comfy_Utils] Warning: Could not import NV_Video_Loader_Path
[NV_Comfy_Utils] NV_Video_Loader_Path NOT registered (import failed)
```

## Step 2: Most Common Issues and Fixes

### Issue 1: VideoHelperSuite Not Installed

**Symptoms:**
- Console shows: `Could not import VideoHelperSuite`
- The folder `ComfyUI/custom_nodes/comfyui-videohelpersuite/` doesn't exist

**Fix:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite comfyui-videohelpersuite
# Then restart ComfyUI
```

### Issue 2: VideoHelperSuite Wrong Folder Name

**Symptoms:**
- VideoHelperSuite is installed but with a different folder name
- Console shows: `VideoHelperSuite path does not exist`

**Fix:**
The folder MUST be named exactly `comfyui-videohelpersuite` (all lowercase). If it's named differently, rename it:
```bash
cd ComfyUI/custom_nodes
# If you have it named differently:
mv ComfyUI-VideoHelperSuite comfyui-videohelpersuite
# Then restart ComfyUI
```

### Issue 3: Python Path Issues

**Symptoms:**
- VideoHelperSuite is installed correctly
- But still shows import errors

**Fix:**
Try reinstalling VideoHelperSuite's requirements:
```bash
cd ComfyUI/custom_nodes/comfyui-videohelpersuite
pip install -r requirements.txt
# Then restart ComfyUI
```

### Issue 4: Cache/Old Files

**Symptoms:**
- Everything looks correct but node still won't work

**Fix:**
1. Stop ComfyUI completely
2. Clear Python cache:
```bash
cd ComfyUI/custom_nodes/NV_Comfy_Utils
find . -type d -name __pycache__ -exec rm -rf {} +
# On Windows PowerShell:
Get-ChildItem -Path . -Directory -Filter __pycache__ -Recurse | Remove-Item -Recurse -Force
```
3. Restart ComfyUI

## Step 3: Verify Installation

Run this checklist:

1. ✅ **VideoHelperSuite exists:**
   - Path: `ComfyUI/custom_nodes/comfyui-videohelpersuite/`
   - Check for file: `comfyui-videohelpersuite/videohelpersuite/load_video_nodes.py`

2. ✅ **NV_Comfy_Utils updated:**
   - Path: `ComfyUI/custom_nodes/NV_Comfy_Utils/`
   - Check for file: `NV_Comfy_Utils/src/KNF_Utils/video_loader_with_frame_replacement.py`

3. ✅ **Git pulled latest changes:**
   ```bash
   cd ComfyUI/custom_nodes/NV_Comfy_Utils
   git pull
   ```

4. ✅ **ComfyUI fully restarted:**
   - Not just refreshed the browser
   - Fully stopped and restarted the Python server

## Step 4: Detailed Debugging

If the above doesn't work, get detailed error information:

1. Start ComfyUI
2. Copy ALL error messages from the console that mention:
   - `[NV_Video_Loader_Path]`
   - `[NV_Comfy_Utils]`
   - `VideoHelperSuite`
   - Any Python tracebacks (lines starting with `Traceback`)

3. Look for the specific error in the traceback

## Common Error Patterns and Solutions

### Error: "ModuleNotFoundError: No module named 'videohelpersuite'"
**Solution:** VideoHelperSuite not installed or wrong folder name

### Error: "AttributeError: ... has no attribute ..."
**Solution:** VideoHelperSuite version too old, update it:
```bash
cd ComfyUI/custom_nodes/comfyui-videohelpersuite
git pull
```

### Error: Node appears in search but clicking does nothing (no console error)
**Solution:** This is usually a frontend cache issue:
1. Hard refresh browser: `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac)
2. Clear browser cache
3. Try incognito/private browsing window

## Step 5: Test VideoHelperSuite Directly

To verify VideoHelperSuite works on its own:

1. Add a standard VHS node to your workflow
2. Try using "Load Video (Path)" from VideoHelperSuite
3. If that doesn't work, the issue is with VideoHelperSuite itself, not NV Video Loader

## Still Not Working?

Provide these details for further help:

1. Operating System (Windows/Mac/Linux)
2. Python version: `python --version`
3. Console output when starting ComfyUI (especially lines with [NV_Video_Loader_Path])
4. Full traceback of any errors
5. Output of:
   ```bash
   ls ComfyUI/custom_nodes/comfyui-videohelpersuite/
   ls ComfyUI/custom_nodes/NV_Comfy_Utils/src/KNF_Utils/
   ```

## Quick Test Script

Create a file `test_import.py` in `ComfyUI/` and run it:

```python
import sys
from pathlib import Path

custom_nodes = Path("custom_nodes")
vhs_path = custom_nodes / "comfyui-videohelpersuite"

print(f"VHS path exists: {vhs_path.exists()}")
print(f"VHS path: {vhs_path.absolute()}")

if vhs_path.exists():
    sys.path.insert(0, str(vhs_path))
    try:
        from videohelpersuite.load_video_nodes import load_video
        print("✅ VideoHelperSuite imports successfully!")
    except Exception as e:
        print(f"❌ VideoHelperSuite import failed: {e}")
else:
    print("❌ VideoHelperSuite directory not found")
```

Run with: `python test_import.py`

