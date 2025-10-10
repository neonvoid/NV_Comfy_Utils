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

### **3. Implement Defensive Programming**
- **Multiple event handlers** for reliability
- **Error handling** for robustness
- **User feedback** for clarity
- **Validation** for safety

### **4. Focus on User Experience**
- **Immediate feedback** for all actions
- **Clear error messages** for debugging
- **Intuitive interface** for ease of use
- **Consistent behavior** across ComfyUI versions

This guide provides the foundation for creating robust, user-friendly ComfyUI custom nodes that work reliably across different environments and ComfyUI versions.
