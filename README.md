# NV_Comfy_Utils - NodeBypasser

A ComfyUI custom node that allows you to bypass/enable multiple nodes by name using a simple text interface.

## 🎯 Features

- **List All Nodes**: View all nodes in your current ComfyUI workflow
- **Text-Based Selection**: Type node names separated by commas to select which nodes to bypass/enable
- **Smart Matching**: Case-insensitive matching that works with both node types and titles
- **Bulk Operations**: Bypass or enable multiple nodes at once
- **Real-time Feedback**: See exactly which nodes were processed and which weren't found

## 🚀 How to Use

### 1. Add the Node
- Search for "NodeBypasser" in ComfyUI's node search
- Add it to your workflow

### 2. List Available Nodes
- Click the **"List All Nodes"** button
- View all nodes in the Status area with their IDs, types, and current state

### 3. Select Nodes to Bypass/Enable
- In the **"Node Names to Bypass"** text field, type the names of nodes you want to affect
- Separate multiple names with commas
- Examples:
  ```
  LoadImage, LoadVideo
  test1, test2
  KNF_Organizer
  ```

### 4. Execute Actions
- Click **"Bypass Nodes"** to bypass all matching nodes
- Click **"Enable Nodes"** to enable all matching nodes
- Check the Status area for results

## 📝 Node Name Examples

Based on your workflow, you can use:

| What to Type | Matches | Result |
|--------------|---------|--------|
| `test1` | Node 50 (LoadVideo - test1) | Bypasses the first LoadVideo node |
| `test2` | Node 51 (LoadVideo - test2) | Bypasses the second LoadVideo node |
| `LoadVideo` | Node 50 & 51 (both LoadVideo nodes) | Bypasses both LoadVideo nodes |
| `LoadImage` | Node 44 (LoadImage) | Bypasses the LoadImage node |
| `KNF_Organizer` | Node 47 (KNF_Organizer) | Bypasses the KNF_Organizer node |

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
└── src/KNF_Utils/
    ├── __init__.py            # Python node exports
    └── nodes.py               # Python nodes (KNF_Organizer, etc.)
```

## 🎨 Widget Types Used

| Widget Type | Purpose | Example |
|-------------|---------|---------|
| `BOOLEAN` | Toggle buttons | "List All Nodes", "Bypass Nodes" |
| `STRING` | Text input | "Node Names to Bypass", "Status" |
| `COMBO` | Dropdown selection | (Not used in current version) |

## 🚨 Troubleshooting

### Node Not Appearing
- Ensure `WEB_DIRECTORY = "./web"` is set in `__init__.py`
- Restart ComfyUI completely
- Check browser console for errors

### Widgets Not Showing
- Add a new NodeBypasser node (don't reuse old ones)
- Check console for widget creation messages
- Verify all widgets are created successfully

### Bypassing Not Working
- Check that node names are typed correctly
- Use "List All Nodes" to see exact node names
- Check the Status area for error messages

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