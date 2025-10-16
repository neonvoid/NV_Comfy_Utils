# 🔘 Momentary Button Node

A frontend-only node that outputs an incrementing INT value each time you press the button.

## Features

- ✅ **No Python backend required** - pure JavaScript implementation
- ✅ **Visual feedback** - button flashes when pressed
- ✅ **State persistence** - counter value saved with workflow
- ✅ **Read-only display** - shows current value, prevents manual editing
- ✅ **Automatic wrapping** - counter wraps at 1,000,000 to prevent overflow

## How to Use

### Basic Usage

1. **Add Node**: Right-click canvas → `NV_Utils` → `Momentary Button`
2. **Press Button**: Click the "TRIGGER" button
3. **Connect Output**: Connect the `trigger` output (INT) to any node that accepts INT input

### Example 1: Force Workflow Re-execution

```
[Momentary Button] → (trigger) → [Any Node]
```

Each button press increments the output value, which forces connected nodes to re-execute even if other inputs haven't changed.

### Example 2: Manual Seed Control

```
[Momentary Button] → (trigger) → [KSampler seed input]
```

Each press generates a new seed value for image generation.

### Example 3: Batch Counter

```
[Momentary Button] → (trigger) → [Custom Counter Logic]
```

Use the incrementing value as a batch index or loop counter.

## Technical Details

- **Output Type**: INT (0 to 999,999)
- **Starting Value**: 0
- **Increment**: +1 per press
- **Wrapping**: Resets to 0 after 999,999
- **Flash Duration**: 150ms
- **Flash Color**: `#5a7a9f` (blue-gray)

## Comparison with Other Nodes

| Feature | Momentary Button | Toggle (Boolean) | Counter Widget |
|---------|------------------|------------------|----------------|
| Output Type | INT (incrementing) | BOOLEAN (true/false) | INT (static) |
| Interaction | Press once | Click to toggle | Manual edit |
| Auto-increment | ✅ Yes | ❌ No | ❌ No |
| Visual feedback | ✅ Flash | ❌ No | ❌ No |
| Force re-execution | ✅ Yes | ⚠️ Only on change | ❌ No |

## Tips

- **Force fresh results**: Connect to nodes that cache results to force re-computation
- **Debugging workflows**: Use to manually step through workflow execution
- **Random seed generation**: Connect directly to seed inputs for easy variation
- **Batch processing**: Use the counter value to track iterations

## Troubleshooting

**Q: Button doesn't seem to trigger execution**  
A: Make sure the downstream node actually uses the INT input. Some nodes cache results.

**Q: Counter resets when I reload the workflow**  
A: This is a bug - the counter should persist. Check browser console for errors.

**Q: Can I reset the counter manually?**  
A: Not directly. Delete and re-add the node to start from 0 again.

**Q: Can I set a custom starting value?**  
A: Currently no, but you can modify `_triggerValue` in the JS file if needed.

## File Location

`custom_nodes/NV_Comfy_Utils/web/momentary_button.js`

## Related Nodes

- `NodeBypasser` - Enable/disable nodes based on boolean inputs
- `SimpleLinkSwitcher` - Switch between multiple input connections
- Built-in `PrimitiveNode` - For static INT values

