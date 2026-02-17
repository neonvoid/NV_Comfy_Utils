# Variables System Guide

Console-driven variable management inspired by Unreal Engine's Blueprint variable system. Variables are created, deleted, and renamed through the Variables Panel. Data sources are assigned via right-click "Promote to Variable" on node outputs. Getter nodes are placed via drag-and-drop from the panel onto the canvas.

## Quick Start

1. **Open the panel**: Click the "Vars" button in the top menu bar, or press `Ctrl+Shift+V`
2. **Create a variable**: Click the `+` button in the panel header, enter a name and optionally select a type
3. **Assign a data source**: Right-click any node on the canvas, select **Promote to Variable**, pick the output slot, then choose your variable
4. **Place a getter**: Drag a variable row from the panel onto the canvas (empty area or directly onto a node's input slot)

## Architecture

The system is entirely frontend-only (no Python backend). Four JS modules work together:

```
variable_manager.js    Central singleton API — all CRUD operations
simple_variables.js    SetVariableNode (hidden) + GetVariableNode (visible)
variable_context_menu.js   "Promote to Variable" right-click menu
variables_panel.js     Panel UI — create, delete, rename, drag-and-drop
```

### How It Works Under the Hood

- A hidden `SetVariableNode` exists in the graph for each variable. It is positioned offscreen at `[-5000, n*60]` and is invisible to the user (draw methods suppressed, `isPointInside` returns false).
- `GetVariableNode` overrides `getInputLink()` to return the link from the matching setter's input. This is how ComfyUI's execution engine resolves data flow through the variable indirection.
- Both node types are registered with `LiteGraph.registerNodeType()` (required for workflow serialization/deserialization), but `SetVariableNode` is removed from link-release menus and has no Python backend definition, so users cannot create it manually.

### Type Resolution Priority

When determining the data type of a variable:

1. **Explicit type** (set by user in panel via "Change Type") takes top priority
2. **Connection-inferred type** (from the source node's output slot) is used as fallback
3. **Wildcard (`*`)** is the default when neither of the above applies

## Panel Features

### Panel Layout

```
+----------------------------------+
| Variables          [+] [R] [^] [x] |  Header: create, refresh, collapse, hide
+----------------------------------+
| Filter variables...               |  Search bar
+----------------------------------+
|  > * my_model     MODEL       :  |  Type-colored dot, name, type badge, menu
|  > * my_latent    LATENT      :  |  Draggable rows
|  > o unassigned   --          :  |  Hollow dot = no source
+----------------------------------+
|  Orphans (1)                      |  Getters with no matching setter
|    ! broken_ref  -- no setter  -> |
+----------------------------------+
```

- **Filled dot**: Variable has a connected source
- **Hollow dot**: Variable exists but has no source assigned
- **Type badge**: Shows the resolved type with color coding

### Creating Variables

Click `+` in the header. An inline input appears with:
- **Name field**: Enter the variable name (duplicates rejected)
- **Type dropdown**: Optionally set an explicit data type. Core ComfyUI types appear first, followed by any custom node types discovered at runtime.

### Drag-and-Drop

Drag any variable row from the panel:
- **Drop on empty canvas**: Creates a `GetVariableNode` at the drop position
- **Drop on a node's input slot**: Creates a `GetVariableNode` and auto-connects it to that input

### Row Context Menu

Click the `...` button on any variable row:

| Action | Description |
|--------|-------------|
| Go to Source | Navigates the canvas to the source node feeding this variable |
| Select All Getters | Multi-selects all getter nodes for this variable on the canvas |
| Rename | Inline rename (updates all setter/getter widgets atomically) |
| Change Type | Set or change the explicit data type |
| Unassign Source | Disconnects the setter from its source node |
| Delete Variable | Removes the setter and ALL getter nodes (with confirmation) |

### Expand Row Details

Click the chevron on a row to expand and see:
- **Source info**: Which node and output slot feeds this variable
- **Getter list**: All getter nodes with navigate buttons
- **Duplicate setter warnings** (if any)

## "Promote to Variable" Context Menu

Right-click any node on the canvas to see this option:

- **Single output node**: Menu shows `Promote to Variable (output_name)` directly with a submenu of existing variables + `+ New Variable...`
- **Multi-output node**: Menu shows `Promote to Variable` with a slot picker submenu first, then the variable list

Selecting an existing variable reassigns that variable's source. Selecting `+ New Variable...` opens a dialog to name a new variable and immediately assign it.

## Type System

### Core Types

The following types are always available:

`*`, `IMAGE`, `LATENT`, `CONDITIONING`, `MODEL`, `CLIP`, `VAE`, `MASK`, `STRING`, `INT`, `FLOAT`, `BOOLEAN`, `CONTROL_NET`, `CLIP_VISION`, `CLIP_VISION_OUTPUT`, `STYLE_MODEL`, `GLIGEN`, `UPSCALE_MODEL`, `SIGMAS`, `NOISE`, `SAMPLER`, `GUIDER`

### Custom Node Types

Types from custom nodes (Impact Pack, KJNodes, etc.) are automatically discovered at runtime by scanning `LiteGraph.slot_types_default_in`, `LiteGraph.slot_types_default_out`, and `LiteGraph.registered_node_types`. They appear below a separator in type dropdowns.

### Type Colors

Known types have curated colors (IMAGE=blue, LATENT=purple, MODEL=green, CONDITIONING=orange, etc.). Unknown/custom types get a deterministic HSL color generated from a hash of the type name, ensuring consistent colors across sessions.

## Backward Compatibility

- **Old workflows with manually-placed SetVariableNodes**: On load, the `onConfigure` hook and `migrateExistingSetters()` detect these nodes, set `_nv_managed = true`, reposition them offscreen, and suppress rendering. The panel adopts them seamlessly.
- **Workflow serialization**: Both node types are registered with LiteGraph, so save/load works normally. Hidden setters serialize their position, properties, and connections like any other node.

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Shift+V` | Toggle Variables Panel visibility |

## Debugging

The variable manager singleton is exposed on `window.NVVariableManager` for console debugging:

```js
// List all variables
NVVariableManager.getAllVariables()

// Get a specific variable's info
NVVariableManager.getVariable("my_image")

// List available types (including custom node types)
NVVariableManager.getVariableTypes()

// Force panel refresh
NVVariablesPanel.getInstance().refresh()
```

## File Locations

| File | Path |
|------|------|
| Variable Manager API | `web/variable_manager.js` |
| Node Definitions | `web/simple_variables.js` |
| Context Menu | `web/variable_context_menu.js` |
| Panel UI | `web/variables_panel.js` |

## Related Features

- **NodeBypasser** (`web/node_bypasser.js`) — Enable/disable nodes; can work alongside variables
- **SimpleLinkSwitcher** (`web/simple_link_switcher.js`) — Switch between input connections
- **Floating Panel** (`web/floating_panel.js`) — Draggable panel infrastructure used by other panels
