# Node Bypasser - Bug Fix Notes

## Issue Fixed
The bypass_input wasn't properly detecting and responding to momentary button pulses.

## Root Causes Identified

### 1. **Reading Order Issue**
- Previously checked widget values BEFORE getOutputData()
- Momentary buttons use `getOutputData()` method for real-time state
- Widget values can be stale when buttons pulse quickly

### 2. **False Value Rejection**
- Previous code: `if (value !== undefined && value !== null)`
- This was correct, but caused confusion
- Changed to: `if (value !== undefined)` for clarity
- Now explicitly accepts `false` as a valid boolean value

## Changes Made

### Updated Methods
1. `getBypassValue()` - Now prioritizes `getOutputData()` over widget reading
2. `getEnableValue()` - Same fix for consistency
3. `getOverrideValue()` - Same fix for consistency

### Reading Priority (new order)
1. **First**: Try `originNode.getOutputData(slot)` - for real-time values
2. **Second**: Try widget named "value", "boolean", or type "toggle"
3. **Third**: Fallback to first widget
4. **Default**: Return false (for override) or button value (for bypass/enable)

## Testing Instructions

### 1. Open Browser Console
- Press F12 in your browser
- Go to Console tab
- You should see detailed logging when buttons are pressed

### 2. Test Bypass Input
```
Setup:
- Connect Momentary Button (Pulse mode) → bypass_input
- Put "LoadImage" (or any node name) in Node Names field
- Click TRIGGER on the momentary button

Expected Console Output:
[NodeBypasser] Reading bypass from getOutputData: true (node: NV/MomentaryButton)
[NodeBypasser 1] Bypass triggered! Override=false
[NodeBypasser] bypassNodesByName called with bypass= true override= false

Expected Result:
- Status should show "Bypassed X nodes"
- Target nodes should turn red (bypassed)
```

### 3. Test Enable Input
```
Setup:
- Connect Momentary Button (Pulse mode) → enable_input
- Click TRIGGER on the momentary button

Expected Console Output:
[NodeBypasser] Reading enable from getOutputData: true (node: NV/MomentaryButton)
[NodeBypasser 1] Enable triggered! Override=false

Expected Result:
- Status should show "Enabled X nodes"
- Target nodes should return to normal color
```

### 4. Test Override Feature
```
Setup:
- Connect Momentary Button → bypass_input
- Connect Boolean Primitive (set to true) → override_input
- Put "LoadImage" in Node Names field
- Put "SaveImage" in Override Node Names field
- Click TRIGGER

Expected Result:
- Status should show "Bypassed X nodes (with override)"
- Both LoadImage AND SaveImage should be bypassed
```

## Troubleshooting

### "No value being read"
**Check**:
- Is the momentary button in "Pulse" mode?
- Is the connection properly made (wire visible)?
- Check console for "Reading bypass from..." messages

### "Bypass happens once then stops"
**This is expected behavior!** The bypasser only triggers on TRUE pulses:
- When button pressed: false → true (triggers)
- When pulse ends: true → false (no action, just resets)
- When button pressed again: false → true (triggers again)

### "Enable works but bypass doesn't"
**Check**:
- Make sure you're connected to `bypass_input`, not `enable_input`
- Both inputs are visually similar - hover to see the name
- Top button input = bypass
- Second button input = enable
- Third button input = override

## Node Names Format

Both main and override node lists support:
- **Exact/Partial**: `LoadImage` matches LoadImage nodes
- **Wildcard**: `Load*` matches anything starting with Load
- **Exclusion**: `!SaveImage` matches everything except SaveImage
- **Multiple**: `LoadImage, SaveImage, Preview*` (comma-separated)

## Selector System

The selector input allows routing bypass/enable commands to specific bypassers:
1. Set different "Selector ID" values on multiple bypassers
2. Connect an INT primitive to selector inputs
3. Only the bypasser with matching ID will respond to triggers
4. Leave selector disconnected for bypasser to work independently

## Visual Feedback

### Node Colors
- **Green tint** (`#224422`): This bypasser is actively selected
- **Dark gray** (`#222222`): Selector active but pointing to different ID
- **Normal**: No selector connected

### Status Messages
- Shows number of nodes affected
- Lists affected nodes by ID and type
- Shows "[Main]" and "[Override]" labels
- Reports any nodes not found

