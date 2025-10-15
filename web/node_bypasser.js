import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const MODE_ALWAYS = 0;
const MODE_BYPASS = 4;

// Create a simple node class that extends LGraphNode
class NodeBypasser extends LGraphNode {
    constructor(title = "Node Bypasser") {
        super(title);
        this.comfyClass = "NodeBypasser";
        this.isVirtualNode = true;  // Mark as virtual to avoid execution errors
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
            default: "Enter node names separated by commas (e.g., LoadImage, LoadVideo, singlePass*, !extension)",
            multiline: true
        }], app).widget;
        this.nodeNamesInput.name = "Node Names to Bypass";
        
        // Add bypass button
        this.bypassButton = ComfyWidgets["BOOLEAN"](this, "bypass_nodes", ["BOOLEAN", { default: false }], app).widget;
        this.bypassButton.name = "Bypass Nodes";
        
        // Add boolean input for bypass trigger
        this.addInput("bypass_input", "BOOLEAN");
        
        // Add enable button
        this.enableButton = ComfyWidgets["BOOLEAN"](this, "enable_nodes", ["BOOLEAN", { default: false }], app).widget;
        this.enableButton.name = "Enable Nodes";
        
        // Add boolean input for enable trigger
        this.addInput("enable_input", "BOOLEAN");
        
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
    
    // Save widget values when they change
    serialize() {
        const data = super.serialize();
        if (data) {
            data.widget_values = {
                node_names: this.nodeNamesInput.value
            };
        }
        return data;
    }
    
    // Load widget values when node is loaded
    configure(info) {
        super.configure(info);
        if (info.widget_values) {
            if (info.widget_values.node_names !== undefined) {
                this.nodeNamesInput.value = info.widget_values.node_names;
            }
        }
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
    
    // Get the effective bypass value (from input or widget)
    getBypassValue() {
        // Check if bypass_input is connected (input slot 0)
        if (this.inputs[0] && this.inputs[0].link != null) {
            const link = this.graph.links[this.inputs[0].link];
            if (link) {
                const originNode = this.graph.getNodeById(link.origin_id);
                if (originNode) {
                    // Try to get value from widget (works for primitive nodes and most boolean sources)
                    if (originNode.widgets && originNode.widgets.length > 0) {
                        const widget = originNode.widgets.find(w => w.type === "toggle" || w.name === "boolean" || w.name === "value");
                        if (widget && widget.value !== undefined) {
                            return widget.value;
                        }
                        if (originNode.widgets[0].value !== undefined) {
                            return originNode.widgets[0].value;
                        }
                    }
                    // Try getOutputData as fallback
                    if (originNode.outputs && originNode.outputs[link.origin_slot]) {
                        const value = originNode.getOutputData ? originNode.getOutputData(link.origin_slot) : null;
                        if (value !== undefined && value !== null) {
                            return value;
                        }
                    }
                }
            }
        }
        return this.bypassButton.value;
    }
    
    // Get the effective enable value (from input or widget)
    getEnableValue() {
        // Check if enable_input is connected (input slot 1)
        if (this.inputs[1] && this.inputs[1].link != null) {
            const link = this.graph.links[this.inputs[1].link];
            if (link) {
                const originNode = this.graph.getNodeById(link.origin_id);
                if (originNode) {
                    // Try to get value from widget (works for primitive nodes and most boolean sources)
                    if (originNode.widgets && originNode.widgets.length > 0) {
                        const widget = originNode.widgets.find(w => w.type === "toggle" || w.name === "boolean" || w.name === "value");
                        if (widget && widget.value !== undefined) {
                            return widget.value;
                        }
                        if (originNode.widgets[0].value !== undefined) {
                            return originNode.widgets[0].value;
                        }
                    }
                    // Try getOutputData as fallback
                    if (originNode.outputs && originNode.outputs[link.origin_slot]) {
                        const value = originNode.getOutputData ? originNode.getOutputData(link.origin_slot) : null;
                        if (value !== undefined && value !== null) {
                            return value;
                        }
                    }
                }
            }
        }
        return this.enableButton.value;
    }
    
    // Check inputs on every draw (this works for virtual nodes)
    onDrawBackground(ctx) {
        this.checkInputs();
    }
    
    // Check the input values and trigger actions
    checkInputs() {
        if (!this.graph) return;
        
        const bypassValue = this.getBypassValue();
        const enableValue = this.getEnableValue();
        
        // Update widgets to reflect input values (visual feedback)
        if (this.inputs[0] && this.inputs[0].link != null) {
            this.bypassButton.value = bypassValue;
        }
        if (this.inputs[1] && this.inputs[1].link != null) {
            this.enableButton.value = enableValue;
        }
        
        // Check if bypass trigger is true and hasn't been processed yet
        if (bypassValue === true && this._lastBypassState !== true) {
            console.log("[NodeBypasser] Bypass activated via input!");
            this.bypassNodesByName(true);
        }
        this._lastBypassState = bypassValue;
        
        // Check if enable trigger is true and hasn't been processed yet
        if (enableValue === true && this._lastEnableState !== true) {
            console.log("[NodeBypasser] Enable activated via input!");
            this.bypassNodesByName(false);
        }
        this._lastEnableState = enableValue;
    }
    
    // Also check on connection changes
    onConnectionsChange(type, index, connected, link_info) {
        // Small delay to ensure data is available
        setTimeout(() => {
            this.checkInputs();
        }, 50);
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
                
                // If this is a subgraph, list internal nodes too
                if (node.subgraph && node.subgraph._nodes) {
                    for (const internalNode of node.subgraph._nodes) {
                        const internalMode = internalNode.mode || 0;
                        let internalModeText = "Active";
                        if (internalMode === MODE_BYPASS) {
                            internalModeText = "Bypassed";
                        } else if (internalMode === 2) {
                            internalModeText = "Muted";
                        }
                        nodeDetails.push(`  └─ Internal ${internalNode.id}: ${internalNode.type} - ${internalNode.title} (${internalModeText})`);
                    }
                }
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
            const errors = [];
            
            for (const nodeName of nodeNames) {
                try {
                    let matchingNodes = [];
                    
                    // Check if it's a regex pattern (contains * or ! or other regex chars)
                    if (this.isRegexPattern(nodeName)) {
                        console.log("[NodeBypasser] Processing regex pattern:", nodeName);
                        matchingNodes = this.findNodesByRegex(nodeName, nodes);
                    } else {
                        // Use simple string matching for non-regex patterns
                        matchingNodes = nodes.filter(node => 
                            node !== this && // Don't include the bypasser node itself
                            (node.type.toLowerCase().includes(nodeName.toLowerCase()) ||
                             (node.title && node.title.toLowerCase().includes(nodeName.toLowerCase())) ||
                             (node.properties && node.properties.name && node.properties.name.toLowerCase().includes(nodeName.toLowerCase())) ||
                             (node.properties && node.properties.variable_name && node.properties.variable_name.toLowerCase().includes(nodeName.toLowerCase())) ||
                             (node.properties && node.properties.custom_name && node.properties.custom_name.toLowerCase().includes(nodeName.toLowerCase())) ||
                             (node._stableCustomName && node._stableCustomName.toLowerCase().includes(nodeName.toLowerCase())))
                        );
                    }
                    
                    if (matchingNodes.length === 0) {
                        notFound.push(nodeName);
                        continue;
                    }
                    
                    for (const targetNode of matchingNodes) {
                        const newMode = bypass ? MODE_BYPASS : MODE_ALWAYS;
                        targetNode.mode = newMode;
                        processedCount++;
                        results.push(`Node ${targetNode.id}: ${targetNode.type}`);
                        
                        // If this is a collapsed subgraph, also bypass all internal nodes
                        if (targetNode.subgraph && targetNode.subgraph._nodes) {
                            console.log(`[NodeBypasser] Subgraph detected in node ${targetNode.id}, bypassing internal nodes`);
                            for (const internalNode of targetNode.subgraph._nodes) {
                                internalNode.mode = newMode;
                                processedCount++;
                                results.push(`  └─ Internal: ${internalNode.id} (${internalNode.type})`);
                            }
                        }
                    }
                } catch (error) {
                    console.error(`[NodeBypasser] Error processing pattern "${nodeName}":`, error);
                    errors.push(`${nodeName}: ${error.message}`);
                }
            }
            
            const action = bypass ? "Bypassed" : "Enabled";
            let resultText = `${action} ${processedCount} nodes:\n${results.join('\n')}`;
            if (notFound.length > 0) {
                resultText += `\n\nNot found: ${notFound.join(', ')}`;
            }
            if (errors.length > 0) {
                resultText += `\n\nErrors: ${errors.join(', ')}`;
            }
            
            this.resultWidget.value = resultText;
            
        } catch (error) {
            console.error('[NodeBypasser] Error bypassing nodes:', error);
            this.resultWidget.value = `Error: ${error.message}`;
        }
    }
    
    // Check if a pattern contains regex-like characters
    isRegexPattern(pattern) {
        // Check for common regex indicators
        return pattern.includes('*') || 
               pattern.includes('!') || 
               pattern.includes('^') || 
               pattern.includes('$') || 
               pattern.includes('[') || 
               pattern.includes(']') || 
               pattern.includes('(') || 
               pattern.includes(')') ||
               pattern.includes('+') ||
               pattern.includes('?') ||
               pattern.includes('|');
    }
    
    // Find nodes using regex patterns
    findNodesByRegex(pattern, nodes) {
        try {
            let regexPattern = pattern;
            let isExclusion = false;
            
            // Handle exclusion patterns (starting with !)
            if (pattern.startsWith('!')) {
                isExclusion = true;
                regexPattern = pattern.substring(1);
            }
            
            // Convert wildcard patterns to regex
            // * becomes .* (match any characters)
            // ? becomes . (match single character)
            regexPattern = regexPattern
                .replace(/\*/g, '.*')
                .replace(/\?/g, '.');
            
            // Create case-insensitive regex
            const regex = new RegExp(regexPattern, 'i');
            
            console.log(`[NodeBypasser] Converted pattern "${pattern}" to regex: /${regexPattern}/i`);
            
            const matchingNodes = nodes.filter(node => {
                if (node === this) return false; // Don't include the bypasser node itself
                
                const typeMatch = regex.test(node.type);
                const titleMatch = node.title ? regex.test(node.title) : false;
                const customNameMatch = node.properties && node.properties.custom_name ? regex.test(node.properties.custom_name) : false;
                const stableCustomNameMatch = node._stableCustomName ? regex.test(node._stableCustomName) : false;
                const matches = typeMatch || titleMatch || customNameMatch || stableCustomNameMatch;
                
                // For exclusion patterns, return the opposite
                return isExclusion ? !matches : matches;
            });
            
            console.log(`[NodeBypasser] Found ${matchingNodes.length} nodes matching pattern "${pattern}"`);
            return matchingNodes;
            
        } catch (error) {
            console.error(`[NodeBypasser] Regex error for pattern "${pattern}":`, error);
            throw new Error(`Invalid regex pattern: ${error.message}`);
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