import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const MODE_ALWAYS = 0;
const MODE_BYPASS = 4;

// Create a simple node class that extends LGraphNode
class NodeBypasser extends LGraphNode {
    constructor(title = "Node Bypasser") {
        super(title);
        this.comfyClass = "NodeBypasser";
        this.isVirtualNode = false;
        this.removed = false;
        this.configuring = false;
        this._tempWidth = 0;
        this.__constructed__ = false;
        this.widgets = this.widgets || [];
        this.properties = this.properties || {};
        
        // Initialize size first to prevent the error
        this.size = [250, 250];
        
        // Add a button to list all nodes
        this.listNodesButton = ComfyWidgets["BOOLEAN"](this, "list_nodes", ["BOOLEAN", { default: false }], app).widget;
        this.listNodesButton.name = "List All Nodes";
        console.log("[NodeBypasser] Created listNodesButton:", this.listNodesButton);
        
        // Add a text input for node names to bypass
        this.nodeNamesInput = ComfyWidgets["STRING"](this, "node_names", ["STRING", { 
            default: "Enter node names separated by commas (e.g., LoadImage, LoadVideo)",
            multiline: true
        }], app).widget;
        this.nodeNamesInput.name = "Node Names to Bypass";
        
        // Add bypass button
        this.bypassButton = ComfyWidgets["BOOLEAN"](this, "bypass_nodes", ["BOOLEAN", { default: false }], app).widget;
        this.bypassButton.name = "Bypass Nodes";
        
        // Add enable button
        this.enableButton = ComfyWidgets["BOOLEAN"](this, "enable_nodes", ["BOOLEAN", { default: false }], app).widget;
        this.enableButton.name = "Enable Nodes";
        
        // Add a result display widget
        this.resultWidget = ComfyWidgets["STRING"](this, "result", ["STRING", { multiline: true }], app).widget;
        this.resultWidget.name = "Status";
        this.resultWidget.inputEl.readOnly = true;
        
        // Set up widget change detection
        const originalOnWidgetChange = this.onWidgetChange;
        this.onWidgetChange = function (widget, value) {
            if (originalOnWidgetChange) {
                originalOnWidgetChange.apply(this, [widget, value]);
            }
            
            // Handle list nodes button
            if (widget === this.listNodesButton && value === true) {
                console.log("[NodeBypasser] List nodes button clicked!");
                this.listAllNodes();
                setTimeout(() => {
                    this.listNodesButton.value = false;
                }, 100);
            }
            
            // Handle bypass button
            if (widget === this.bypassButton && value === true) {
                this.bypassNodesByName(true);
                setTimeout(() => {
                    this.bypassButton.value = false;
                }, 100);
            }
            
            // Handle enable button
            if (widget === this.enableButton && value === true) {
                this.bypassNodesByName(false);
                setTimeout(() => {
                    this.enableButton.value = false;
                }, 100);
            }
        };
        
        // Add backup onClick handlers for debugging
        setTimeout(() => {
            console.log("[NodeBypasser] Setting up backup onClick handlers...");
            
            if (this.listNodesButton && this.listNodesButton.onClick) {
                console.log("[NodeBypasser] Adding onClick to list button");
                const originalOnClick = this.listNodesButton.onClick;
                this.listNodesButton.onClick = (options) => {
                    console.log("[NodeBypasser] List button onClick triggered!");
                    this.listAllNodes();
                    if (originalOnClick) {
                        originalOnClick.call(this.listNodesButton, options);
                    }
                };
            } else {
                console.log("[NodeBypasser] List button or onClick not found:", this.listNodesButton);
            }
            
            if (this.bypassButton && this.bypassButton.onClick) {
                console.log("[NodeBypasser] Adding onClick to bypass button");
                const originalBypassOnClick = this.bypassButton.onClick;
                this.bypassButton.onClick = (options) => {
                    console.log("[NodeBypasser] Bypass button onClick triggered!");
                    this.bypassNodesByName(true);
                    if (originalBypassOnClick) {
                        originalBypassOnClick.call(this.bypassButton, options);
                    }
                };
            } else {
                console.log("[NodeBypasser] Bypass button or onClick not found:", this.bypassButton);
            }
            
            if (this.enableButton && this.enableButton.onClick) {
                console.log("[NodeBypasser] Adding onClick to enable button");
                const originalEnableOnClick = this.enableButton.onClick;
                this.enableButton.onClick = (options) => {
                    console.log("[NodeBypasser] Enable button onClick triggered!");
                    this.bypassNodesByName(false);
                    if (originalEnableOnClick) {
                        originalEnableOnClick.call(this.enableButton, options);
                    }
                };
            } else {
                console.log("[NodeBypasser] Enable button or onClick not found:", this.enableButton);
            }
        }, 1000);
        
        this.onConstructed();
    }
    
    onConstructed() {
        this.__constructed__ = true;
        // Ensure size is set after construction
        this.size = [250, 250];
        console.log("[NodeBypasser] Node constructed, widgets:", this.widgets.length);
        console.log("[NodeBypasser] Widget names:", this.widgets.map(w => w.name));
    }
    
    computeSize() {
        // Make sure size is always defined
        if (!this.size || this.size.length !== 2) {
            this.size = [250, 250];
        }
        return this.size;
    }
    
    listAllNodes() {
        try {
            console.log("[NodeBypasser] listAllNodes() called");
            
            const graph = app.graph;
            console.log("[NodeBypasser] Graph:", graph);
            if (!graph) {
                this.resultWidget.value = "Error: No graph found";
                return;
            }
            
            const nodes = graph._nodes;
            console.log("[NodeBypasser] Nodes:", nodes);
            console.log("[NodeBypasser] Number of nodes:", nodes ? nodes.length : "undefined");
            
            if (!nodes) {
                this.resultWidget.value = "Error: No nodes found in graph";
                return;
            }
            
            let nodeDetails = [];
            
            // List all nodes
            for (const node of nodes) {
                const nodeId = node.id;
                const nodeType = node.type || 'Unknown';
                const nodeTitle = node.title || 'No title';
                const nodeMode = node.mode || 0;
                
                console.log(`[NodeBypasser] Processing node: ${nodeId} - ${nodeType} - ${nodeTitle}`);
                
                // Skip the bypasser node itself
                if (node === this) {
                    console.log("[NodeBypasser] Skipping self node");
                    continue;
                }
                
                // Get mode description
                let modeText = "Active";
                if (nodeMode === MODE_BYPASS) {
                    modeText = "Bypassed";
                } else if (nodeMode === 2) {
                    modeText = "Muted";
                }
                
                nodeDetails.push(`Node ${nodeId}: ${nodeType} - ${nodeTitle} (${modeText})`);
            }
            
            console.log("[NodeBypasser] Node details:", nodeDetails);
            
            if (nodeDetails.length === 0) {
                this.resultWidget.value = "No other nodes found in graph";
                return;
            }
            
            const result = `Found ${nodeDetails.length} nodes in graph:\n\n${nodeDetails.join('\n')}`;
            this.resultWidget.value = result;
            console.log("[NodeBypasser] Result set:", result);
            
        } catch (error) {
            console.error('[NodeBypasser] Error listing nodes:', error);
            this.resultWidget.value = `Error: ${error.message}`;
        }
    }
    
    // Bypass nodes by name
    bypassNodesByName(bypass) {
        try {
            console.log("[NodeBypasser] bypassNodesByName called with bypass=", bypass);
            
            const nodeNamesText = this.nodeNamesInput.value;
            console.log("[NodeBypasser] Node names text:", nodeNamesText);
            
            if (!nodeNamesText || nodeNamesText.trim() === "" || nodeNamesText.includes("Enter node names")) {
                this.resultWidget.value = "Please enter node names to bypass";
                return;
            }
            
            // Parse the comma-separated node names
            const nodeNames = nodeNamesText.split(',').map(name => name.trim()).filter(name => name.length > 0);
            
            const graph = app.graph;
            if (!graph) {
                this.resultWidget.value = "Error: No graph found";
                return;
            }
            
            const nodes = graph._nodes;
            let processedCount = 0;
            const results = [];
            const notFound = [];
            
            for (const nodeName of nodeNames) {
                // Find nodes that match the name (case-insensitive)
                const matchingNodes = nodes.filter(node => 
                    node !== this && // Don't include the bypasser node itself
                    (node.type.toLowerCase().includes(nodeName.toLowerCase()) ||
                     (node.title && node.title.toLowerCase().includes(nodeName.toLowerCase())))
                );
                
                if (matchingNodes.length === 0) {
                    notFound.push(nodeName);
                    continue;
                }
                
                for (const targetNode of matchingNodes) {
                    const newMode = bypass ? MODE_BYPASS : MODE_ALWAYS;
                    targetNode.mode = newMode;
                    processedCount++;
                    results.push(`Node ${targetNode.id}: ${targetNode.type}`);
                }
            }
            
            const action = bypass ? "Bypassed" : "Enabled";
            let resultText = `${action} ${processedCount} nodes:\n${results.join('\n')}`;
            if (notFound.length > 0) {
                resultText += `\n\nNot found: ${notFound.join(', ')}`;
            }
            
            this.resultWidget.value = resultText;
            
        } catch (error) {
            console.error('[NodeBypasser] Error bypassing nodes:', error);
            this.resultWidget.value = `Error: ${error.message}`;
        }
    }
        
}

// Set up the node properties
NodeBypasser.type = "NodeBypasser";
NodeBypasser.title = "Node Bypasser";
NodeBypasser.category = "KNF_Utils";
NodeBypasser.description = "Bypass nodes by name";

// Register the extension
console.log("[NodeBypasser] Extension loading...");
app.registerExtension({
    name: "NV_Comfy_Utils.NodeBypasser",
    registerCustomNodes() {
        console.log("[NodeBypasser] Registering custom nodes...");
        LiteGraph.registerNodeType(NodeBypasser.type, NodeBypasser);
        console.log("[NodeBypasser] Node type registered:", NodeBypasser.type);
    },
    loadedGraphNode(node) {
        if (node.type == "NodeBypasser") {
            console.log("[NodeBypasser] Node loaded in graph:", node.id);
            node._tempWidth = node.size[0];
        }
    },
});
console.log("[NodeBypasser] Extension registered");