# ComfyUI Custom Node Development Guide for LLM Agents

## 🎯 **Purpose**
This guide provides comprehensive context for LLM agents developing ComfyUI custom nodes, combining official documentation with real-world implementation experience.

## 📚 **Official Documentation Reference**
Based on [ComfyUI Custom Nodes Overview](https://docs.comfy.org/custom-nodes/overview)

## 🏗️ **ComfyUI Architecture Overview**

### **Client-Server Model**
ComfyUI operates on a **client-server architecture**:
- **Server (Python)**: Handles data processing, AI models, image diffusion, file I/O
- **Client (JavaScript)**: Manages user interface, node interactions, real-time updates
- **API Mode**: Server can be used independently without the client

### **Four Categories of Custom Nodes**

#### 1. **Server Side Only** (Most Common)
- **Purpose**: Data processing, AI operations, file handling
- **Implementation**: Python class with input/output types and processing function
- **Communication**: Through ComfyUI's data flow system
- **Example**: Image processors, model loaders, file operations

#### 2. **Client Side Only** (Our NodeBypasser)
- **Purpose**: UI modifications, real-time interactions, state management
- **Implementation**: JavaScript extending LGraphNode
- **Communication**: Direct access to `app.graph` and live workflow state
- **Example**: Node bypassers, UI helpers, workflow organizers

#### 3. **Independent Client and Server**
- **Purpose**: Separate server features with related UI enhancements
- **Implementation**: Both Python backend and JavaScript frontend
- **Communication**: Through ComfyUI's data flow control
- **Example**: Custom data types with specialized widgets

#### 4. **Connected Client and Server**
- **Purpose**: Direct client-server communication
- **Implementation**: Custom communication protocols
- **Limitation**: **Not compatible with API mode**
- **Example**: Real-time status updates, custom protocols

## 🚀 **Frontend-Only Node Development (Client Side Only)**

### **When to Use Frontend-Only Nodes**
- **Real-time UI interactions** (bypassing, toggling, organizing)
- **Immediate feedback** requirements
- **State manipulation** without data processing
- **Workflow management** tools
- **No server-side processing** needed

### **Core Implementation Pattern**

```javascript
// 1. Import ComfyUI modules
import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// 2. Define constants
const MODE_ALWAYS = 0;
const MODE_BYPASS = 4;

// 3. Create node class extending LGraphNode
class CustomNode extends LGraphNode {
    constructor(title = "Custom Node") {
        super(title);
        
        // 4. Set up node properties
        this.comfyClass = "CustomNode";
        this.isVirtualNode = false;
        this.removed = false;
        this.configuring = false;
        this._tempWidth = 0;
        this.__constructed__ = false;
        this.widgets = this.widgets || [];
        this.properties = this.properties || {};
        
        // 5. Initialize size
        this.size = [250, 250];
        
        // 6. Create widgets
        this.createWidgets();
        
        // 7. Set up event handling
        this.setupEventHandlers();
        
        this.onConstructed();
    }
    
    createWidgets() {
        // Create interactive widgets
        this.actionButton = ComfyWidgets["BOOLEAN"](this, "action", ["BOOLEAN", { default: false }], app).widget;
        this.actionButton.name = "Action Button";
        
        this.textInput = ComfyWidgets["STRING"](this, "text", ["STRING", { default: "Enter text" }], app).widget;
        this.textInput.name = "Text Input";
        
        this.resultWidget = ComfyWidgets["STRING"](this, "result", ["STRING", { multiline: true }], app).widget;
        this.resultWidget.name = "Result";
        this.resultWidget.inputEl.readOnly = true;
    }
    
    setupEventHandlers() {
        // Primary: onWidgetChange approach
        const originalOnWidgetChange = this.onWidgetChange;
        this.onWidgetChange = function (widget, value) {
            if (originalOnWidgetChange) {
                originalOnWidgetChange.apply(this, [widget, value]);
            }
            
            if (widget === this.actionButton && value === true) {
                this.performAction();
                setTimeout(() => {
                    this.actionButton.value = false;
                }, 100);
            }
        };
        
        // Backup: Direct onClick handlers (for reliability)
        setTimeout(() => {
            if (this.actionButton && this.actionButton.onClick) {
                const originalOnClick = this.actionButton.onClick;
                this.actionButton.onClick = (options) => {
                    this.performAction();
                    if (originalOnClick) {
                        originalOnClick.call(this.actionButton, options);
                    }
                };
            }
        }, 1000);
    }
    
    performAction() {
        // Access live graph state
        const graph = app.graph;
        const nodes = graph._nodes;
        
        // Perform actions on nodes
        for (const node of nodes) {
            if (node !== this) {
                // Modify node properties
                node.mode = MODE_BYPASS;
            }
        }
        
        // Update UI
        this.resultWidget.value = "Action completed";
    }
    
    onConstructed() {
        this.__constructed__ = true;
        this.size = [250, 250];
    }
    
    computeSize() {
        if (!this.size || this.size.length !== 2) {
            this.size = [250, 250];
        }
        return this.size;
    }
}

// 8. Set up node properties
CustomNode.type = "CustomNode";
CustomNode.title = "Custom Node";
CustomNode.category = "Custom";
CustomNode.description = "Custom node description";

// 9. Register the extension
app.registerExtension({
    name: "CustomNode.Extension",
    registerCustomNodes() {
        LiteGraph.registerNodeType(CustomNode.type, CustomNode);
    },
    loadedGraphNode(node) {
        if (node.type == "CustomNode") {
            node._tempWidth = node.size[0];
        }
    },
});
```

## 🔧 **Key Technical Concepts**

### **Widget System**
```javascript
// Widget types and their uses
ComfyWidgets["BOOLEAN"]()    // Toggle buttons, checkboxes
ComfyWidgets["STRING"]()     // Text inputs, multiline text
ComfyWidgets["COMBO"]()      // Dropdown selections
ComfyWidgets["NUMBER"]()     // Numeric inputs
ComfyWidgets["INT"]()        // Integer inputs
ComfyWidgets["FLOAT"]()      // Float inputs
```

### **Graph Access**
```javascript
// Access live workflow state
const graph = app.graph;           // Current workflow
const nodes = graph._nodes;        // All nodes in workflow
const links = graph._links;        // All connections between nodes

// Node properties
node.id          // Unique node identifier
node.type        // Node type (e.g., "LoadImage")
node.title       // Display name
node.mode        // Execution mode (0=always, 4=bypass, 2=muted)
node.pos         // Position [x, y]
node.size        // Size [width, height]
```

### **Reading Input Values from Connected Nodes**

When your node has input slots that receive values from other nodes (like primitive boolean or integer nodes), you need to read their values properly:

```javascript
// CORRECT: Find input by name (robust, cross-machine compatible)
getInputValue(inputName) {
    const inputSlot = this.inputs.findIndex(i => i.name === inputName);
    
    if (inputSlot >= 0 && this.inputs[inputSlot].link != null) {
        const link = this.graph.links[this.inputs[inputSlot].link];
        if (link) {
            const originNode = this.graph.getNodeById(link.origin_id);
            if (originNode && originNode.widgets && originNode.widgets.length > 0) {
                // Try to find the specific widget type
                const widget = originNode.widgets.find(w => 
                    w.type === "toggle" || w.type === "number" || 
                    w.name === "boolean" || w.name === "value" || w.name === "int"
                );
                if (widget && widget.value !== undefined) {
                    return widget.value;
                }
                // Fallback to first widget
                if (originNode.widgets[0].value !== undefined) {
                    return originNode.widgets[0].value;
                }
            }
        }
    }
    return null; // No connection or no value
}

// Usage example:
const selectorValue = this.getInputValue("selector");
const bypassValue = this.getInputValue("bypass_input");
const enableValue = this.getInputValue("enable_input");
```

**⚠️ WRONG: Using hardcoded slot indices**
```javascript
// DON'T DO THIS - fragile, breaks across machines/versions
if (this.inputs[0].link != null) { /* ... */ }  // ❌ Hardcoded index
if (this.inputs[1].link != null) { /* ... */ }  // ❌ Hardcoded index
```

**Why `findIndex` by name is critical:**
- Input order can vary across different ComfyUI versions
- Workflow serialization may reorder inputs
- Adding/removing inputs changes indices
- Cross-machine compatibility requires name-based lookup

### **Node Modes**
```javascript
const MODE_ALWAYS = 0;    // Node executes normally
const MODE_BYPASS = 4;    // Node is bypassed (skipped)
const MODE_MUTED = 2;     // Node is muted (different from bypass)
```

### **Event Handling Patterns**

#### **Primary: onWidgetChange**
```javascript
this.onWidgetChange = function (widget, value) {
    if (widget === this.button && value === true) {
        this.doAction();
    }
};
```
**Pros**: Standard ComfyUI approach
**Cons**: Sometimes unreliable with BOOLEAN widgets

#### **Backup: Direct onClick**
```javascript
this.button.onClick = (options) => {
    this.doAction();
};
```
**Pros**: More reliable, direct control
**Cons**: Bypasses ComfyUI's event system

#### **Recommended: Both Approaches**
Use both for maximum compatibility across ComfyUI versions.

## 📁 **File Structure**

```
CustomNode/
├── __init__.py                 # Entry point with WEB_DIRECTORY
├── web/
│   ├── extensions.js           # Frontend entry point
│   └── custom_node.js         # Main implementation
└── src/
    └── CustomNode/
        ├── __init__.py        # Python exports (if needed)
        └── nodes.py           # Python nodes (if needed)
```

### **Required Files**

#### **__init__.py (Root)**
```python
import os

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__author__ = "Your Name"
__email__ = "your@email.com"
__version__ = "0.0.1"

# For frontend-only nodes, these can be empty
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Critical: Tell ComfyUI where to find frontend files
WEB_DIRECTORY = "./web"
```

#### **web/extensions.js**
```javascript
// Import all frontend extensions
import "./custom_node.js";
```

## 🎯 **Development Best Practices**

### **1. Widget Creation**
- **Initialize size early** to prevent errors
- **Set meaningful names** for widgets
- **Use appropriate widget types** for the use case
- **Handle multiline text** for status displays

### **2. Event Handling**
- **Use both onWidgetChange and onClick** for reliability
- **Reset button values** after actions
- **Handle errors gracefully**
- **Provide user feedback**

### **3. Graph Manipulation**
- **Always check for graph existence**
- **Exclude self from operations**
- **Handle node not found cases**
- **Provide clear error messages**

### **4. User Experience**
- **Immediate feedback** for all actions
- **Clear status messages**
- **Error handling with explanations**
- **Intuitive widget labels**

## 🚨 **Common Pitfalls and Solutions**

### **1. Widget Click Detection Issues**
**Problem**: Buttons don't respond to clicks
**Solution**: Use both onWidgetChange and onClick handlers

### **2. Size Initialization Errors**
**Problem**: "newSize is undefined" errors
**Solution**: Initialize `this.size = [width, height]` early in constructor

### **3. Graph Access Issues**
**Problem**: Cannot access `app.graph`
**Solution**: Ensure you're in the correct context, check for graph existence

### **4. Widget Not Appearing**
**Problem**: Node appears in search but widgets don't show
**Solution**: Check WEB_DIRECTORY path, verify file structure

### **5. Event Handler Timing**
**Problem**: onClick handlers not working
**Solution**: Use setTimeout to ensure widgets are fully initialized

### **6. State Tracking Issues with Selector/Switch Inputs**
**Problem**: After changing selector/switch input, need to press bypass/enable multiple times for it to take effect

**Root Cause**: State tracking variables (`_lastBypassState`, `_lastTriggerState`) persist across selector changes. When you:
1. Set `bypass_input = true` with selector `1` → NodeBypasser #1 activates, sets `_lastBypassState = true`
2. Change selector to `2` → `bypass_input` is still `true` (hasn't changed)
3. NodeBypasser #2 checks: `bypassValue === true && this._lastBypassState !== true` → **FALSE** (already true)
4. Action doesn't trigger! ❌

**Solution**: Track selector/switch changes and reset state when they change:

```javascript
checkInputs() {
    const bypassValue = this.getBypassValue();
    const selectorValue = this.getSelectorValue();
    const myId = this.selectorIdWidget.value;
    
    // CRITICAL: Detect selector changes and reset state tracking
    if (selectorValue !== this._lastSelectorValue) {
        console.log(`[Node ${myId}] Selector changed from ${this._lastSelectorValue} to ${selectorValue}`);
        // Reset state tracking so bypass/enable can trigger again
        this._lastBypassState = undefined;
        this._lastEnableState = undefined;
        this._lastSelectorValue = selectorValue;
    }
    
    const isSelectorActive = selectorValue !== null && selectorValue === myId;
    
    // Now check if action should trigger
    if (bypassValue === true && this._lastBypassState !== true) {
        if (selectorValue === null || isSelectorActive) {
            console.log(`[Node ${myId}] Bypass activated!`);
            this.performBypass();
            this._lastBypassState = true;
        } else {
            this._lastBypassState = bypassValue; // Track to avoid spam
        }
    } else if (bypassValue === false || bypassValue === null) {
        this._lastBypassState = bypassValue;
    }
}
```

**Key Points**:
- Always track the control input (selector/switch) separately
- Reset dependent state variables when control input changes
- This allows actions to re-trigger even if action inputs haven't changed
- Applies to any pattern with multiple control inputs (selector + bypass/enable/trigger)

### **7. Off-Screen Node Input Processing**
**Problem**: Node doesn't process inputs when scrolled off-screen or in a collapsed workflow area

**Root Cause**: `onDrawBackground()` only runs when the node is actively being drawn/rendered. If the node is:
- Scrolled outside the visible viewport
- In a collapsed group
- On a hidden layer
Then `onDrawBackground()` won't be called, so `checkInputs()` never runs! ❌

**Solution**: Add a timer-based check that runs regardless of visibility:

```javascript
onConstructed() {
    this.__constructed__ = true;
    this.size = [250, 250];
    
    // CRITICAL: Set up timer to check inputs even when node is off-screen
    // This ensures inputs are processed regardless of viewport visibility
    this._inputCheckInterval = setInterval(() => {
        if (this.graph && !this.removed) {
            this.checkInputs();
        }
    }, 100); // Check every 100ms
}

onRemoved() {
    // IMPORTANT: Clean up interval when node is removed to prevent memory leaks
    if (this._inputCheckInterval) {
        clearInterval(this._inputCheckInterval);
        this._inputCheckInterval = null;
    }
}

// Keep onDrawBackground as well for efficiency when node IS visible
onDrawBackground(ctx) {
    this.checkInputs();
}
```

**Benefits**:
- Node processes inputs even when off-screen
- Works in collapsed groups, hidden layers, etc.
- Timer (100ms) is frequent enough for responsiveness
- Combined with `onDrawBackground` for efficiency when visible
- Proper cleanup prevents memory leaks

**Performance Considerations**:
- 100ms interval = 10 checks per second (very lightweight)
- Only runs when `this.graph` exists and node isn't removed
- More reliable than viewport-dependent methods
- Worth the tiny performance cost for guaranteed input processing

## 🔍 **Debugging Techniques**

### **Console Logging**
```javascript
console.log("[NodeName] Debug message:", data);
```

### **Widget Inspection**
```javascript
console.log("Widget:", this.button);
console.log("Widget onClick:", this.button.onClick);
```

### **Graph State Inspection**
```javascript
console.log("Graph:", app.graph);
console.log("Nodes:", app.graph._nodes);
console.log("Node count:", app.graph._nodes.length);
```

### **Manual Testing**
```javascript
// Test in browser console
app.graph._nodes.forEach(node => {
    if (node.type === "YourNodeType") {
        node.yourMethod();
    }
});
```

## 📖 **Advanced Patterns**

### **Regex Pattern Matching**
```javascript
isRegexPattern(pattern) {
    return pattern.includes('*') || pattern.includes('!') || 
           pattern.includes('^') || pattern.includes('$');
}

findNodesByRegex(pattern, nodes) {
    let regexPattern = pattern;
    let isExclusion = pattern.startsWith('!');
    
    if (isExclusion) {
        regexPattern = pattern.substring(1);
    }
    
    regexPattern = regexPattern
        .replace(/\*/g, '.*')
        .replace(/\?/g, '.');
    
    const regex = new RegExp(regexPattern, 'i');
    
    return nodes.filter(node => {
        if (node === this) return false;
        const matches = regex.test(node.type) || 
                       (node.title && regex.test(node.title));
        return isExclusion ? !matches : matches;
    });
}
```

### **Multi-Widget Operations**
```javascript
// Handle multiple widgets with different actions
this.onWidgetChange = function (widget, value) {
    if (widget === this.button1 && value === true) {
        this.action1();
    } else if (widget === this.button2 && value === true) {
        this.action2();
    } else if (widget === this.toggle && value !== undefined) {
        this.toggleAction(value);
    }
};
```

### **Selector Pattern for Multiple Nodes**

When you have multiple instances of the same node type and want to control which one is active via a single integer input:

```javascript
constructor() {
    super();
    // Add a selector ID widget (unique identifier for this instance)
    this.selectorIdWidget = ComfyWidgets["INT"](this, "selector_id", 
        ["INT", { default: 0, min: 0, max: 999, step: 1 }], app).widget;
    this.selectorIdWidget.name = "Selector ID";
    
    // Add selector input to receive the active ID
    this.addInput("selector", "INT");
    
    // Add your action inputs
    this.addInput("bypass_input", "BOOLEAN");
    this.addInput("enable_input", "BOOLEAN");
}

checkInputs() {
    // Get the current selector value (which ID is active)
    const selectorValue = this.getInputValue("selector"); // Use name-based lookup!
    const myId = this.selectorIdWidget.value;
    
    // Check if this node is the active one
    const isSelectorActive = selectorValue !== null && selectorValue === myId;
    
    // Visual feedback
    if (isSelectorActive) {
        this.bgcolor = "#224422"; // Green tint for active
    } else if (selectorValue !== null) {
        this.bgcolor = "#222222"; // Dark gray for inactive
    } else {
        this.bgcolor = null; // Normal when no selector connected
    }
    
    // Only process actions if this node is active OR no selector is connected
    const bypassValue = this.getInputValue("bypass_input");
    if (bypassValue === true) {
        if (selectorValue === null || isSelectorActive) {
            console.log(`[Node ${myId}] Bypass activated!`);
            this.performBypass();
        } else {
            console.log(`[Node ${myId}] Bypass ignored - selector mismatch`);
        }
    }
}
```

**Use case**: Have 6 `NodeBypasser` instances with IDs 0-5. Connect a single INT primitive to all their `selector` inputs. Change the INT value to activate different bypassers.

**Benefits**:
- Clean workflow switching without complex routing
- Single control point for multiple configurations
- Visual feedback shows which node is active
- Graceful degradation when selector is disconnected

### **Error Handling and User Feedback**
```javascript
try {
    // Perform action
    const result = this.performAction();
    this.resultWidget.value = `Success: ${result}`;
} catch (error) {
    console.error('[NodeName] Error:', error);
    this.resultWidget.value = `Error: ${error.message}`;
}
```

## 🔄 **Working with ComfyUI's Native Subgraphs**

### **Understanding Subgraph Types**

ComfyUI has **two different features** both called "subgraphs" - don't confuse them:

#### **1. Frontend Collapsed Subgraphs** (Visual Organization)
- **Purpose**: Group existing nodes into a collapsible container
- **Created by**: User selects nodes → "Convert to Subgraph"
- **Visibility**: Container node in main graph, internal nodes hidden visually
- **Execution**: Internal nodes still execute independently
- **Type**: Frontend-only visual feature

#### **2. Backend Dynamic Subgraphs** (Node Expansion)
- **Purpose**: Nodes that create other nodes at runtime
- **Created by**: Node returns `{"expand": graph, "result": outputs}`
- **Uses**: `GraphBuilder` from `comfy_execution.graph_utils`
- **Execution**: Ephemeral nodes added during execution
- **Type**: Backend feature for advanced workflows

### **Frontend Collapsed Subgraphs Structure**

When you collapse nodes into a subgraph, ComfyUI creates this structure:

```javascript
// Container node in main graph
const containerNode = {
    id: 7,
    type: "5c031040-f269-4237-9f7e-48fc80229b97",  // UUID type
    title: "New Subgraph",
    graph: { ... },           // Outer graph (what you see)
    subgraph: {               // THIS is where internal nodes are!
        _nodes: [             // Array of actual internal nodes
            { id: 1, type: "LoadImage", ... },
            { id: 2, type: "VHS_DuplicateImages", ... },
            { id: 6, type: "PreviewImage", ... }
        ]
    }
}
```

**Key insight**: `node.subgraph._nodes` contains the actual working nodes!

### **Working with Collapsed Subgraphs**

#### **Detecting Subgraph Nodes**
```javascript
function isSubgraphNode(node) {
    // Check if node has internal subgraph structure
    return node.subgraph && node.subgraph._nodes && node.subgraph._nodes.length > 0;
}
```

#### **Getting Internal Nodes**
```javascript
function getInternalNodes(subgraphNode) {
    if (!subgraphNode.subgraph || !subgraphNode.subgraph._nodes) {
        return [];
    }
    return subgraphNode.subgraph._nodes;
}
```

#### **Bypassing Subgraph and Contents**
```javascript
function bypassSubgraph(subgraphNode, bypass) {
    const newMode = bypass ? MODE_BYPASS : MODE_ALWAYS;
    
    // 1. Bypass the container
    subgraphNode.mode = newMode;
    
    // 2. Bypass all internal nodes (CRITICAL!)
    if (subgraphNode.subgraph && subgraphNode.subgraph._nodes) {
        for (const internalNode of subgraphNode.subgraph._nodes) {
            internalNode.mode = newMode;
        }
    }
}
```

### **Why This Matters**

**Problem**: Bypassing only the container node doesn't stop internal nodes from executing!

```javascript
// ❌ WRONG - Internal nodes still execute
containerNode.mode = MODE_BYPASS;

// ✅ CORRECT - Bypass container AND internal nodes
containerNode.mode = MODE_BYPASS;
if (containerNode.subgraph && containerNode.subgraph._nodes) {
    for (const internalNode of containerNode.subgraph._nodes) {
        internalNode.mode = MODE_BYPASS;
    }
}
```

### **Listing Subgraph Contents**

```javascript
function listAllNodes() {
    const nodeDetails = [];
    
    for (const node of app.graph._nodes) {
        // List the container
        nodeDetails.push(`Node ${node.id}: ${node.type}`);
        
        // List internal nodes with tree structure
        if (node.subgraph && node.subgraph._nodes) {
            for (const internalNode of node.subgraph._nodes) {
                nodeDetails.push(`  └─ Internal ${internalNode.id}: ${internalNode.type}`);
            }
        }
    }
    
    return nodeDetails;
}
```

**Output example:**
```
Node 7: 5c031040-f269-4237-9f7e-48fc80229b97
  └─ Internal 1: LoadImage
  └─ Internal 2: VHS_DuplicateImages
  └─ Internal 6: PreviewImage
```

### **Complete Subgraph-Aware Node Bypasser**

```javascript
bypassNodesByName(nodeName, bypass) {
    const matchingNodes = this.findNodesByName(nodeName);
    const newMode = bypass ? MODE_BYPASS : MODE_ALWAYS;
    let processedCount = 0;
    
    for (const targetNode of matchingNodes) {
        // Bypass the node
        targetNode.mode = newMode;
        processedCount++;
        
        // If it's a collapsed subgraph, also bypass internal nodes
        if (targetNode.subgraph && targetNode.subgraph._nodes) {
            console.log(`[NodeBypasser] Subgraph detected, bypassing ${targetNode.subgraph._nodes.length} internal nodes`);
            
            for (const internalNode of targetNode.subgraph._nodes) {
                internalNode.mode = newMode;
                processedCount++;
            }
        }
    }
    
    return processedCount;
}
```

### **Subgraph Node ID Structure**

- **Container node**: Regular integer ID (e.g., `7`)
- **Node type**: UUID format (e.g., `"5c031040-f269-4237-9f7e-48fc80229b97"`)
- **Internal nodes**: Have their own IDs within the subgraph
- **Execution**: Internal nodes execute in the main workflow, not isolated

### **Testing Subgraph Detection**

```javascript
// In browser console
let subgraphNodes = app.graph._nodes.filter(n => 
    n.subgraph && n.subgraph._nodes
);

console.log("Found subgraphs:", subgraphNodes.length);
subgraphNodes.forEach(sg => {
    console.log(`Subgraph ${sg.id}:`, sg.title);
    console.log(`  Contains ${sg.subgraph._nodes.length} nodes`);
    sg.subgraph._nodes.forEach(n => {
        console.log(`    - ${n.id}: ${n.type}`);
    });
});
```

### **Common Subgraph Pitfalls**

#### **1. Not Bypassing Internal Nodes**
**Problem**: Bypassing container but internal nodes still execute
**Solution**: Always iterate `node.subgraph._nodes` and bypass them too

#### **2. Confusing Subgraph Types**
**Problem**: Mixing up frontend collapsed subgraphs with backend dynamic subgraphs
**Solution**: 
- Frontend: Uses `node.subgraph._nodes`
- Backend: Uses `comfy_execution.graph_utils.GraphBuilder`

#### **3. Assuming Container Controls Execution**
**Problem**: Thinking container bypass automatically bypasses contents
**Solution**: Internal nodes are independent - you must explicitly bypass them

#### **4. Missing Edge Cases**
**Problem**: Not checking if subgraph structure exists
**Solution**: Always check: `if (node.subgraph && node.subgraph._nodes)`

### **Best Practices for Subgraph Handling**

1. **Always check for subgraph structure** before accessing internal nodes
2. **Iterate internal nodes** when performing bulk operations
3. **Provide visual feedback** showing internal node operations
4. **Use tree structure** (└─) when displaying nested nodes
5. **Handle both cases** - nodes with and without subgraphs

---

## 🔘 **Momentary Button Pattern**

Momentary buttons are useful for triggering workflow execution or incrementing counters. Unlike toggle switches, they:
- **Trigger once** per press
- **Output an incrementing value** to force workflow re-execution
- **Provide visual feedback** during the press

### **Implementation Example**

```javascript
function MomentaryButtonNode() {
    this.addOutput("trigger", "INT");
    
    // Internal counter
    this._triggerValue = 0;
    
    // Create button
    this.pressButton = this.addWidget("button", "press_button", null, () => {
        this.onButtonPress();
    });
    this.pressButton.label = "TRIGGER";
    
    // Display widget (read-only)
    this.displayWidget = ComfyWidgets["INT"](this, "current_value", 
        ["INT", { default: 0, min: 0, max: 999999, step: 1 }], app).widget;
    this.displayWidget.callback = () => {
        // Prevent manual changes
        this.displayWidget.value = this._triggerValue;
    };
    
    this.isVirtualNode = true;
}

MomentaryButtonNode.prototype.onButtonPress = function() {
    // Increment trigger value
    this._triggerValue = (this._triggerValue + 1) % 1000000;
    this.displayWidget.value = this._triggerValue;
    
    // Visual feedback - flash the button
    const originalBgColor = this.bgcolor;
    this.bgcolor = "#5a7a9f";
    this.setDirtyCanvas(true, true);
    
    setTimeout(() => {
        this.bgcolor = originalBgColor;
        this.setDirtyCanvas(true, true);
    }, 150);
    
    // Mark graph as changed
    if (this.graph) {
        this.graph.change();
    }
};

// Provide output value
MomentaryButtonNode.prototype.onDrawBackground = function(ctx) {
    if (this.outputs && this.outputs[0]) {
        this.setOutputData(0, this._triggerValue);
    }
};

// Save/restore state
MomentaryButtonNode.prototype.serialize = function() {
    return { trigger_value: this._triggerValue };
};

MomentaryButtonNode.prototype.configure = function(data) {
    if (data.trigger_value !== undefined) {
        this._triggerValue = data.trigger_value;
        this.displayWidget.value = this._triggerValue;
    }
};
```

### **Key Features**

1. **Incrementing Output**: Each press increments the output value, which can trigger downstream nodes
2. **Visual Feedback**: Button flashes when pressed (150ms duration)
3. **Read-Only Display**: Shows current value but prevents manual editing
4. **State Persistence**: Saves trigger value when workflow is saved
5. **Graph Change Notification**: Calls `graph.change()` to trigger workflow updates

### **Common Use Cases**

- **Force Workflow Re-execution**: Connect to a node to force it to re-run
- **Increment Counters**: Use in loops or batch processing
- **Manual Triggers**: User-initiated actions in automated workflows
- **Debug Checkpoints**: Pause workflow and trigger specific steps manually

### **Best Practices**

1. **Wrap counter values** using modulo to prevent overflow
2. **Provide visual feedback** so users know the button was pressed
3. **Use short flash duration** (100-200ms) for responsive feel
4. **Mark display widgets as read-only** to prevent confusion
5. **Always call `graph.change()`** to ensure workflow updates

---

## 🎓 **Key Takeaways for LLM Agents**

### **1. Choose the Right Architecture**
- **Frontend-only**: UI interactions, real-time updates, state management
- **Backend-only**: Data processing, AI operations, file I/O
- **Both**: Complex features requiring server processing with UI enhancements

### **2. Understand ComfyUI's Patterns**
- **Widget system** for UI components
- **Event handling** for user interactions
- **Graph access** for workflow manipulation
- **Node modes** for execution control
- **Subgraph structure** for collapsed node groups

### **3. Implement Defensive Programming**
- **Multiple event handlers** for reliability
- **Error handling** for robustness
- **User feedback** for clarity
- **Validation** for safety
- **Subgraph detection** for complete operations

### **4. Focus on User Experience**
- **Immediate feedback** for all actions
- **Clear error messages** for debugging
- **Intuitive interface** for ease of use
- **Consistent behavior** across ComfyUI versions
- **Visual hierarchy** for nested structures

### **5. Handle Subgraphs Correctly**
- **Distinguish between** frontend collapsed subgraphs and backend dynamic subgraphs
- **Always process internal nodes** when operating on subgraph containers
- **Provide feedback** showing both container and internal node operations
- **Check for subgraph structure** before accessing internal nodes

This guide provides the foundation for creating robust, user-friendly ComfyUI custom nodes that work reliably across different environments and ComfyUI versions.
