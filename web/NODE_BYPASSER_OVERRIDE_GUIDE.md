# Node Bypasser - Override Input Guide

## Overview
The Node Bypasser now includes an **override input** feature that acts as a **disable switch** for the bypasser.

## New Feature

### Override Input
- **Type**: Boolean
- **Purpose**: When `true`, blocks all bypass/enable actions
- **Behavior**: Acts as a safety switch to temporarily disable the bypasser

## How It Works

### Basic Behavior
- **Override = false** (or disconnected): Normal operation - bypass/enable inputs work as expected
- **Override = true**: All bypass/enable actions are blocked - triggers are ignored

### Example Use Case
```
Scenario: You have a mode where you don't want certain nodes to be bypassed

Setup:
- Node Names: "LoadImage, SaveImage"
- Connect your mode switch → override_input

When override=false:
  - Momentary button triggers work normally
  - Nodes are bypassed/enabled as expected
  
When override=true:
  - Momentary button triggers are blocked
  - Console shows: "🚫 Bypass blocked by override"
  - Nodes remain in their current state
```

## Inputs

| Input Name | Type | Purpose |
|------------|------|---------|
| `selector` | INT | Route bypass/enable actions to specific bypasser by ID |
| `bypass_input` | BOOLEAN | Trigger to bypass nodes |
| `enable_input` | BOOLEAN | Trigger to enable nodes |
| `override_input` | BOOLEAN | ✨ **NEW** - Determines if override nodes should be affected |

## Configuration

### Node Names (Main Group)
- These nodes are **always** affected when bypass/enable is triggered
- Supports wildcards: `LoadImage*`, `!SaveImage`
- Comma-separated list

### Override Node Names
- These nodes are **only** affected when `override_input` is `true`
- Uses the same pattern matching as main node names
- Comma-separated list

## Visual Feedback

The status widget will show:
- Number of nodes affected
- Whether override was active: `"Bypassed 5 nodes (with override)"`
- Breakdown showing `[Main]` and `[Override]` groups

## Example Connections

### Simple Mode Toggle
```
[Momentary Button] ---> [bypass_input]
[Boolean Primitive] --> [override_input]
```

### Selector-Based Routing
```
[Int Primitive] -----> [selector]
[Momentary Button] --> [bypass_input]
[LazySwitch] --------> [override_input]
```

## Tips

1. **Empty Override List**: If override node names is empty or contains the placeholder text, it won't process anything even when override is true
2. **Manual Buttons**: The manual "Bypass Nodes" and "Enable Nodes" buttons also respect the current override input value
3. **Console Logging**: Check the browser console for detailed logging: `[NodeBypasser] Bypass triggered! Override=true`

## Advanced Pattern Matching

Both main and override node lists support:
- **Exact match**: `LoadImage`
- **Partial match**: `Load` (matches LoadImage, LoadVideo, etc.)
- **Wildcards**: `Load*` (matches anything starting with "Load")
- **Exclusion**: `!SaveImage` (matches everything except SaveImage)
- **Custom names**: Matches node titles, custom_name properties, and _stableCustomName

## Migration from Previous Version

If you're upgrading from a previous version:
1. Your existing node bypasser instances will continue to work as before
2. The override input defaults to `false`, so existing behavior is preserved
3. Simply connect a boolean to the new `override_input` to start using the new feature

