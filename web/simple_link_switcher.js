import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Simple Link Switcher - MVP version for switching between two inputs to one target
class SimpleLinkSwitcher extends LGraphNode {
    constructor(title = "Simple Link Switcher") {
        super(title);
        this.comfyClass = "SimpleLinkSwitcher";
        this.isVirtualNode = true;
        this.removed = false;
        this.configuring = false;
        this._tempWidth = 0;
        this.__constructed__ = false;
        this.widgets = this.widgets || [];
        this.properties = this.properties || {};
        
        console.log("[SimpleLinkSwitcher] Constructor called");
        
        // Initialize size
        this.size = [300, 200];
        
        // Input 1 pattern
        this.input1Widget = ComfyWidgets["STRING"](this, "input1", ["STRING", { 
            default: "Input 1 pattern (e.g., LoadImage,PrimitiveInt)",
            multiline: false
        }], app).widget;
        this.input1Widget.name = "Input 1";
        
        // Input 2 pattern
        this.input2Widget = ComfyWidgets["STRING"](this, "input2", ["STRING", { 
            default: "Input 2 pattern (e.g., LoadVideo,PrimitiveFloat)",
            multiline: false
        }], app).widget;
        this.input2Widget.name = "Input 2";
        
        // Target pattern
        this.targetWidget = ComfyWidgets["STRING"](this, "target", ["STRING", { 
            default: "Target pattern (e.g., SetVariableNode,GetVariableNode)",
            multiline: false
        }], app).widget;
        this.targetWidget.name = "Target";
        
        // Switch between inputs
        this.switchWidget = ComfyWidgets["BOOLEAN"](this, "switch", ["BOOLEAN", { default: false }], app).widget;
        this.switchWidget.name = "Use Input 2";
        
        // Action button
        this.connectButton = ComfyWidgets["BOOLEAN"](this, "connect", ["BOOLEAN", { default: false }], app).widget;
        this.connectButton.name = "Connect";
        
        // Status display
        this.statusWidget = ComfyWidgets["STRING"](this, "status", ["STRING", { multiline: true }], app).widget;
        this.statusWidget.name = "Status";
        this.statusWidget.inputEl.readOnly = true;
        
        // Set up event handlers
        this.setupEventHandlers();
        
        this.onConstructed();
    }
    
    // Save widget values when they change
    serialize() {
        const data = super.serialize();
        if (data) {
            data.widget_values = {
                input1: this.input1Widget.value,
                input2: this.input2Widget.value,
                target: this.targetWidget.value,
                switch: this.switchWidget.value
            };
        }
        return data;
    }
    
    // Load widget values when node is loaded
    configure(info) {
        super.configure(info);
        if (info.widget_values) {
            if (info.widget_values.input1 !== undefined) {
                this.input1Widget.value = info.widget_values.input1;
            }
            if (info.widget_values.input2 !== undefined) {
                this.input2Widget.value = info.widget_values.input2;
            }
            if (info.widget_values.target !== undefined) {
                this.targetWidget.value = info.widget_values.target;
            }
            if (info.widget_values.switch !== undefined) {
                this.switchWidget.value = info.widget_values.switch;
            }
        }
    }
    
    setupEventHandlers() {
        const originalOnWidgetChange = this.onWidgetChange;
        this.onWidgetChange = function (widget, value) {
            if (originalOnWidgetChange) {
                originalOnWidgetChange.apply(this, [widget, value]);
            }
            
            // Handle switch change
            if (widget === this.switchWidget) {
                this.onSwitchChange(value);
            }
            
            // Handle connect button
            if (widget === this.connectButton && value === true) {
                this.connectNodes();
                setTimeout(() => {
                    this.connectButton.value = false;
                }, 100);
            }
        };
        
        // Backup onClick handler
        setTimeout(() => {
            if (this.connectButton && this.connectButton.onClick) {
                const originalOnClick = this.connectButton.onClick;
                this.connectButton.onClick = (options) => {
                    this.connectNodes();
                    if (originalOnClick) {
                        originalOnClick.call(this.connectButton, options);
                    }
                };
            }
        }, 1000);
    }
    
    onConstructed() {
        this.__constructed__ = true;
        this.size = [300, 200];
    }
    
    computeSize() {
        if (!this.size || this.size.length !== 2) {
            this.size = [300, 200];
        }
        return this.size;
    }
    
    onSwitchChange(useInput2) {
        const currentInput = useInput2 ? this.input2Widget.value : this.input1Widget.value;
        const inputCount = currentInput.split(',').length;
        this.statusWidget.value = `Selected: ${currentInput} (${inputCount} nodes)`;
    }
    
    // Find multiple nodes by comma-separated patterns
    findNodesByPattern(pattern, nodes) {
        if (!pattern || pattern.trim() === "") {
            console.log("[SimpleLinkSwitcher] Empty pattern provided");
            return [];
        }
        
        // Split by comma and trim whitespace
        const patterns = pattern.split(',').map(p => p.trim()).filter(p => p.length > 0);
        console.log("[SimpleLinkSwitcher] Searching for patterns:", patterns);
        console.log("[SimpleLinkSwitcher] Available nodes:", nodes.map(n => n.type));
        
        const foundNodes = [];
        
        for (const singlePattern of patterns) {
            console.log("[SimpleLinkSwitcher] Searching for single pattern:", singlePattern);
            
            // Simple exact match first
            let found = nodes.find(node => 
                node !== this &&
                (node.type === singlePattern || 
                 (node.title && node.title === singlePattern))
            );
            
            if (found && !foundNodes.includes(found)) {
                console.log("[SimpleLinkSwitcher] Found exact match:", found.type, "ID:", found.id);
                foundNodes.push(found);
                continue;
            }
            
            // Then try partial matches
            found = nodes.find(node => 
                node !== this &&
                !foundNodes.includes(node) &&
                (node.type.toLowerCase().includes(singlePattern.toLowerCase()) ||
                 (node.title && node.title.toLowerCase().includes(singlePattern.toLowerCase())))
            );
            
            if (found) {
                console.log("[SimpleLinkSwitcher] Found partial match:", found.type, "ID:", found.id);
                foundNodes.push(found);
            } else {
                console.log("[SimpleLinkSwitcher] No match found for pattern:", singlePattern);
                // Let's see what nodes are available for debugging
                const availableTypes = nodes.filter(n => n !== this).map(n => n.type);
                console.log("[SimpleLinkSwitcher] Available node types:", availableTypes);
            }
        }
        
        console.log("[SimpleLinkSwitcher] Total found nodes:", foundNodes.length);
        return foundNodes;
    }
    
    // Connect nodes
    connectNodes() {
        try {
            const graph = app.graph;
            if (!graph) {
                this.statusWidget.value = "Error: No graph found";
                return;
            }
            
            const nodes = graph._nodes;
            
            // Get the selected input pattern
            const useInput2 = this.switchWidget.value;
            const inputPattern = useInput2 ? this.input2Widget.value : this.input1Widget.value;
            const targetPattern = this.targetWidget.value;
            
            console.log("[SimpleLinkSwitcher] Switch state:", useInput2 ? "Input 2" : "Input 1");
            console.log("[SimpleLinkSwitcher] Selected input pattern:", inputPattern);
            
            if (!inputPattern || !targetPattern) {
                this.statusWidget.value = "Error: Please specify both input and target patterns";
                return;
            }
            
            // Find the nodes
            console.log("[SimpleLinkSwitcher] Looking for input nodes with pattern:", inputPattern);
            const inputNodes = this.findNodesByPattern(inputPattern, nodes);
            console.log("[SimpleLinkSwitcher] Found input nodes:", inputNodes.map(n => n.type));
            
            console.log("[SimpleLinkSwitcher] Looking for target nodes with pattern:", targetPattern);
            const targetNodes = this.findNodesByPattern(targetPattern, nodes);
            console.log("[SimpleLinkSwitcher] Found target nodes:", targetNodes.map(n => n.type));
            
            if (inputNodes.length === 0) {
                this.statusWidget.value = `Error: No input nodes found: ${inputPattern}`;
                return;
            }
            
            if (targetNodes.length === 0) {
                this.statusWidget.value = `Error: No target nodes found: ${targetPattern}`;
                return;
            }
            
            // Disconnect any existing connections from target nodes
            for (const targetNode of targetNodes) {
                this.disconnectTargetInput(targetNode);
            }
            
            // Create one-to-one connections between input and target nodes
            let successCount = 0;
            let errorCount = 0;
            const connectionResults = [];
            
            // Map inputs to targets in order
            const maxConnections = Math.min(inputNodes.length, targetNodes.length);
            
            for (let i = 0; i < maxConnections; i++) {
                const inputNode = inputNodes[i];
                const targetNode = targetNodes[i];
                
                console.log("[SimpleLinkSwitcher] Attempting one-to-one connection:", inputNode.type, "ID:", inputNode.id, "->", targetNode.type, "ID:", targetNode.id);
                const success = this.createConnection(inputNode, targetNode);
                if (success) {
                    successCount++;
                    connectionResults.push(`${inputNode.type} -> ${targetNode.type}`);
                    console.log("[SimpleLinkSwitcher] Connection successful");
                } else {
                    errorCount++;
                    console.log("[SimpleLinkSwitcher] Connection failed");
                }
            }
            
            // If there are more inputs than targets, connect remaining inputs to the last target
            if (inputNodes.length > targetNodes.length) {
                const lastTarget = targetNodes[targetNodes.length - 1];
                for (let i = targetNodes.length; i < inputNodes.length; i++) {
                    const inputNode = inputNodes[i];
                    console.log("[SimpleLinkSwitcher] Connecting extra input to last target:", inputNode.type, "ID:", inputNode.id, "->", lastTarget.type, "ID:", lastTarget.id);
                    const success = this.createConnection(inputNode, lastTarget);
                    if (success) {
                        successCount++;
                        connectionResults.push(`${inputNode.type} -> ${lastTarget.type}`);
                        console.log("[SimpleLinkSwitcher] Connection successful");
                    } else {
                        errorCount++;
                        console.log("[SimpleLinkSwitcher] Connection failed");
                    }
                }
            }
            
            // Update status
            if (successCount > 0) {
                this.statusWidget.value = `Connected ${successCount} links:\n${connectionResults.join('\n')}`;
            } else {
                this.statusWidget.value = `Error: Could not create any connections`;
            }
            
        } catch (error) {
            console.error('[SimpleLinkSwitcher] Error connecting nodes:', error);
            this.statusWidget.value = `Error: ${error.message}`;
        }
    }
    
    // Disconnect the first input of a target node
    disconnectTargetInput(targetNode) {
        try {
            if (!targetNode.inputs || targetNode.inputs.length === 0) return;
            
            const firstInput = targetNode.inputs[0];
            if (firstInput && firstInput.link) {
                this.disconnectLink(firstInput.link);
            }
        } catch (error) {
            console.error('[SimpleLinkSwitcher] Error disconnecting target input:', error);
        }
    }
    
    // Create connection between input and target nodes
    createConnection(inputNode, targetNode) {
        try {
            const graph = app.graph;
            if (!graph) return false;
            
            // Find the first available output from input node
            let outputSlot = -1;
            if (inputNode.outputs) {
                for (let i = 0; i < inputNode.outputs.length; i++) {
                    if (inputNode.outputs[i] && inputNode.outputs[i].type) {
                        outputSlot = i;
                        break;
                    }
                }
            }
            
            if (outputSlot === -1) {
                this.statusWidget.value = "Error: Input node has no outputs";
                return false;
            }
            
            // Use the first input slot of target node
            const inputSlot = 0;
            
            // Check if connection is possible
            if (!this.canConnect(inputNode, targetNode, outputSlot, inputSlot)) {
                const inputType = inputNode.outputs[outputSlot].type || 'unknown';
                const targetType = targetNode.inputs[inputSlot].type || 'unknown';
                this.statusWidget.value = `Error: Incompatible types - Input: ${inputType}, Target: ${targetType}`;
                return false;
            }
            
            // Use ComfyUI's built-in connect method
            const success = inputNode.connect(outputSlot, targetNode, inputSlot);
            
            if (success) {
                // Update the graph display
                graph.setDirtyCanvas(true, true);
                return true;
            } else {
                this.statusWidget.value = "Error: Connection failed";
                return false;
            }
            
        } catch (error) {
            console.error('[SimpleLinkSwitcher] Error creating connection:', error);
            return false;
        }
    }
    
    // Check if two nodes can be connected
    canConnect(inputNode, targetNode, outputSlot, inputSlot) {
        console.log("[SimpleLinkSwitcher] Checking connection compatibility:");
        console.log("  Input node:", inputNode.type, "output slot:", outputSlot);
        console.log("  Target node:", targetNode.type, "input slot:", inputSlot);
        
        if (!inputNode.outputs || !targetNode.inputs) {
            console.log("  Error: Missing outputs or inputs");
            return false;
        }
        
        if (!inputNode.outputs[outputSlot] || !targetNode.inputs[inputSlot]) {
            console.log("  Error: Invalid slot numbers");
            return false;
        }
        
        const output = inputNode.outputs[outputSlot];
        const input = targetNode.inputs[inputSlot];
        
        console.log("  Output type:", output.type);
        console.log("  Input type:", input.type);
        
        // If both have types, they should match (unless input accepts any type with "*")
        if (output.type && input.type && input.type !== "*" && output.type !== input.type) {
            console.log("  Error: Type mismatch");
            return false;
        }
        
        console.log("  Connection is valid");
        return true;
    }
    
    // Disconnect a specific link
    disconnectLink(linkId) {
        try {
            const graph = app.graph;
            if (!graph || !graph._links[linkId]) return false;
            
            const link = graph._links[linkId];
            const sourceNode = graph._nodes.find(n => n.id === link.origin_id);
            const targetNode = graph._nodes.find(n => n.id === link.target_id);
            
            // Remove from source node
            if (sourceNode && sourceNode.outputs && sourceNode.outputs[link.origin_slot]) {
                const output = sourceNode.outputs[link.origin_slot];
                if (output.links) {
                    const index = output.links.indexOf(linkId);
                    if (index > -1) {
                        output.links.splice(index, 1);
                    }
                }
            }
            
            // Remove from target node
            if (targetNode && targetNode.inputs && targetNode.inputs[link.target_slot]) {
                targetNode.inputs[link.target_slot].link = null;
            }
            
            // Remove from graph
            delete graph._links[linkId];
            graph.setDirtyCanvas(true, true);
            
            return true;
        } catch (error) {
            console.error('[SimpleLinkSwitcher] Error disconnecting link:', error);
            return false;
        }
    }
}

// Set up the node properties
SimpleLinkSwitcher.type = "SimpleLinkSwitcher";
SimpleLinkSwitcher.title = "Simple Link Switcher";
SimpleLinkSwitcher.category = "KNF_Utils";
SimpleLinkSwitcher.description = "Switch between two input nodes to one target node";

// Make sure the node is searchable
console.log("[SimpleLinkSwitcher] Node properties set:", {
    type: SimpleLinkSwitcher.type,
    title: SimpleLinkSwitcher.title,
    category: SimpleLinkSwitcher.category
});

// Register the extension
console.log("[SimpleLinkSwitcher] Registering extension...");
app.registerExtension({
    name: "NV_Comfy_Utils.SimpleLinkSwitcher",
    registerCustomNodes() {
        console.log("[SimpleLinkSwitcher] Registering node type:", SimpleLinkSwitcher.type);
        LiteGraph.registerNodeType(SimpleLinkSwitcher.type, SimpleLinkSwitcher);
        console.log("[SimpleLinkSwitcher] Node type registered successfully");
    },
    loadedGraphNode(node) {
        if (node.type == "SimpleLinkSwitcher") {
            console.log("[SimpleLinkSwitcher] Node loaded in graph:", node.id);
            node._tempWidth = node.size[0];
        }
    },
});
console.log("[SimpleLinkSwitcher] Extension registered");
