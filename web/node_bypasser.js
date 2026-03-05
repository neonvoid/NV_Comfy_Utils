import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const MODE_ALWAYS = 0;
const MODE_BYPASS = 4;
const NODE_SLOT_HEIGHT = 20;
const NODE_WIDGET_HEIGHT = 20;

// Deferred class definition — LGraphNode must exist before we can extend it
let NodeBypasser = null;

function defineNodeBypasser() {
    if (NodeBypasser) return; // Already defined

    class _NodeBypasser extends LGraphNode {
        constructor(title = "Node Bypasser") {
            super(title);
            this.comfyClass = "NodeBypasser";
            this.isVirtualNode = true;
            this.removed = false;
            this.configuring = false;
            this._tempWidth = 0;
            this.__constructed__ = false;
            this.widgets = this.widgets || [];
            this.properties = this.properties || {};

            // Initialize size
            this.size = [250, 100];

            // Add selector ID widget
            this.selectorIdWidget = ComfyWidgets["INT"](this, "selector_id", ["INT", { default: 0, min: 0, max: 999, step: 1 }], app).widget;
            this.selectorIdWidget.name = "Selector ID";

            // List All Nodes — button widget with direct callback
            this.listNodesButton = this.addWidget("button", "List All Nodes", null, () => {
                this.listAllNodes();
            });
            this.listNodesButton.serialize = false;

            // Add a text input for node names to bypass
            this.nodeNamesInput = ComfyWidgets["STRING"](this, "node_names", ["STRING", {
                default: "Enter node names separated by commas (e.g., LoadImage, singlePass*, !extension, @GroupName)",
                multiline: true
            }], app).widget;
            this.nodeNamesInput.name = "Node Names to Bypass";

            // Bypass button — direct callback
            this.bypassButton = this.addWidget("button", "Bypass Nodes", null, () => {
                this.bypassNodesByName(true);
            });
            this.bypassButton.serialize = false;

            // Add selector input (connects to INT source)
            this.addInput("selector", "INT");

            // Add boolean input for bypass trigger
            this.addInput("bypass_input", "BOOLEAN");

            // Enable button — direct callback
            this.enableButton = this.addWidget("button", "Enable Nodes", null, () => {
                this.bypassNodesByName(false);
            });
            this.enableButton.serialize = false;

            // Add boolean input for enable trigger
            this.addInput("enable_input", "BOOLEAN");

            // Add override boolean input (acts as disable switch)
            this.addInput("override_input", "BOOLEAN");

            // Add a result display widget
            this.resultWidget = ComfyWidgets["STRING"](this, "result", ["STRING", { multiline: true }], app).widget;
            this.resultWidget.name = "Status";
            this.resultWidget.inputEl.readOnly = true;

            this.onConstructed();
        }

        // Save widget values when they change
        serialize() {
            const data = super.serialize();
            if (data) {
                data.widget_values = {
                    node_names: this.nodeNamesInput.value,
                    selector_id: this.selectorIdWidget.value
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
                if (info.widget_values.selector_id !== undefined) {
                    this.selectorIdWidget.value = info.widget_values.selector_id;
                }
            }
        }

        onConstructed() {
            this.__constructed__ = true;
            this.size = this.computeSize();

            // Set up timer to check inputs even when node is off-screen
            this._inputCheckInterval = setInterval(() => {
                if (this.graph && !this.removed) {
                    this.checkInputs();
                }
            }, 50);
        }

        onRemoved() {
            this.removed = true;
            if (this._inputCheckInterval) {
                clearInterval(this._inputCheckInterval);
                this._inputCheckInterval = null;
            }
        }

        computeSize(width) {
            const w = width || this.size?.[0] || 250;
            const numSlots = Math.max(
                (this.inputs ? this.inputs.length : 0),
                (this.outputs ? this.outputs.length : 0)
            );
            const slotHeight = numSlots * NODE_SLOT_HEIGHT;
            const widgetHeight = (this.widgets ? this.widgets.length : 0) * NODE_WIDGET_HEIGHT;
            // Extra padding for multiline STRING widgets (node_names + status)
            const multilineExtra = 80;
            const h = slotHeight + widgetHeight + multilineExtra + 10;
            return [Math.max(w, 250), Math.max(h, 200)];
        }

        // Get the effective bypass value (from input or widget)
        getBypassValue() {
            const bypassInputSlot = this.inputs.findIndex(i => i.name === "bypass_input");

            if (bypassInputSlot >= 0 && this.inputs[bypassInputSlot].link != null) {
                const link = this.graph.links[this.inputs[bypassInputSlot].link];
                if (link) {
                    const originNode = this.graph.getNodeById(link.origin_id);
                    if (originNode) {
                        // Try getOutputData FIRST for momentary button compatibility
                        if (originNode.getOutputData && originNode.outputs && originNode.outputs[link.origin_slot]) {
                            const value = originNode.getOutputData(link.origin_slot);
                            if (value !== undefined) {
                                return value;
                            }
                        }
                        // Try to get value from widget as fallback (works for primitive nodes)
                        if (originNode.widgets && originNode.widgets.length > 0) {
                            const widget = originNode.widgets.find(w => w.type === "toggle" || w.name === "boolean" || w.name === "value");
                            if (widget && widget.value !== undefined) {
                                return widget.value;
                            }
                            if (originNode.widgets[0].value !== undefined) {
                                return originNode.widgets[0].value;
                            }
                        }
                    }
                }
            }
            return false;
        }

        // Get the effective enable value (from input or widget)
        getEnableValue() {
            const enableInputSlot = this.inputs.findIndex(i => i.name === "enable_input");

            if (enableInputSlot >= 0 && this.inputs[enableInputSlot].link != null) {
                const link = this.graph.links[this.inputs[enableInputSlot].link];
                if (link) {
                    const originNode = this.graph.getNodeById(link.origin_id);
                    if (originNode) {
                        if (originNode.getOutputData && originNode.outputs && originNode.outputs[link.origin_slot]) {
                            const value = originNode.getOutputData(link.origin_slot);
                            if (value !== undefined) {
                                return value;
                            }
                        }
                        if (originNode.widgets && originNode.widgets.length > 0) {
                            const widget = originNode.widgets.find(w => w.type === "toggle" || w.name === "boolean" || w.name === "value");
                            if (widget && widget.value !== undefined) {
                                return widget.value;
                            }
                            if (originNode.widgets[0].value !== undefined) {
                                return originNode.widgets[0].value;
                            }
                        }
                    }
                }
            }
            return false;
        }

        // Get the selector value (from input or null if not connected)
        getSelectorValue() {
            const selectorInputSlot = this.inputs.findIndex(i => i.name === "selector");

            if (selectorInputSlot >= 0 && this.inputs[selectorInputSlot].link != null) {
                const link = this.graph.links[this.inputs[selectorInputSlot].link];
                if (link) {
                    const originNode = this.graph.getNodeById(link.origin_id);
                    if (originNode && originNode.widgets && originNode.widgets.length > 0) {
                        const widget = originNode.widgets.find(w =>
                            w.type === "number" || w.name === "value" || w.name === "int"
                        );
                        if (widget && widget.value !== undefined) {
                            return widget.value;
                        }
                        return originNode.widgets[0].value;
                    }
                }
            }
            return null;
        }

        // Get the override value (from input)
        getOverrideValue() {
            const overrideInputSlot = this.inputs.findIndex(i => i.name === "override_input");

            if (overrideInputSlot >= 0 && this.inputs[overrideInputSlot].link != null) {
                const link = this.graph.links[this.inputs[overrideInputSlot].link];
                if (link) {
                    const originNode = this.graph.getNodeById(link.origin_id);
                    if (originNode) {
                        if (originNode.getOutputData && originNode.outputs && originNode.outputs[link.origin_slot]) {
                            const value = originNode.getOutputData(link.origin_slot);
                            if (value !== undefined) {
                                return value;
                            }
                        }
                        if (originNode.widgets && originNode.widgets.length > 0) {
                            const widget = originNode.widgets.find(w => w.type === "toggle" || w.name === "boolean" || w.name === "value");
                            if (widget && widget.value !== undefined) {
                                return widget.value;
                            }
                            if (originNode.widgets[0].value !== undefined) {
                                return originNode.widgets[0].value;
                            }
                        }
                    }
                }
            }
            return false;
        }

        // Check the input values and trigger actions (runs on interval)
        checkInputs() {
            if (!this.graph) return;

            const bypassValue = this.getBypassValue();
            const enableValue = this.getEnableValue();
            const selectorValue = this.getSelectorValue();
            const overrideValue = this.getOverrideValue();
            const myId = this.selectorIdWidget.value;

            const isSelectorActive = selectorValue !== null && selectorValue === myId;

            // Detect selector change — reset state tracking
            if (selectorValue !== this._lastSelectorValue) {
                this._lastBypassState = undefined;
                this._lastEnableState = undefined;
                this._lastSelectorValue = selectorValue;
            }

            // Visual indicator based on selector
            if (isSelectorActive) {
                this.bgcolor = "#224422";
            } else if (selectorValue !== null) {
                this.bgcolor = "#222222";
            } else {
                this.bgcolor = null;
            }

            // Check override
            if (overrideValue === true) {
                if (bypassValue === true && this._lastBypassState !== true) {
                    this._lastBypassState = true;
                } else if (bypassValue !== true && this._lastBypassState === true) {
                    this._lastBypassState = bypassValue;
                }
                if (enableValue === true && this._lastEnableState !== true) {
                    this._lastEnableState = true;
                } else if (enableValue !== true && this._lastEnableState === true) {
                    this._lastEnableState = enableValue;
                }
                return;
            }

            // Check bypass input (only respond to TRUE pulses)
            if (bypassValue === true && this._lastBypassState !== true) {
                if (selectorValue === null || isSelectorActive) {
                    this.bypassNodesByName(true);
                }
                this._lastBypassState = true;
            } else if (bypassValue !== true && this._lastBypassState === true) {
                this._lastBypassState = bypassValue;
            }

            // Check enable input (only respond to TRUE pulses)
            if (enableValue === true && this._lastEnableState !== true) {
                if (selectorValue === null || isSelectorActive) {
                    this.bypassNodesByName(false);
                }
                this._lastEnableState = true;
            } else if (enableValue !== true && this._lastEnableState === true) {
                this._lastEnableState = enableValue;
            }
        }

        listAllNodes() {
            try {
                const graph = app.graph;
                if (!graph) {
                    this.resultWidget.value = "Error: No graph found";
                    return;
                }

                const nodes = graph._nodes;
                if (!nodes) {
                    this.resultWidget.value = "Error: No nodes found in graph";
                    return;
                }

                let nodeDetails = [];

                for (const node of nodes) {
                    if (node === this) continue;

                    const nodeId = node.id;
                    const nodeType = node.type || 'Unknown';
                    const nodeTitle = node.title || 'No title';
                    const nodeMode = node.mode || 0;

                    let modeText = "Active";
                    if (nodeMode === MODE_BYPASS) {
                        modeText = "Bypassed";
                    } else if (nodeMode === 2) {
                        modeText = "Muted";
                    }

                    nodeDetails.push(`Node ${nodeId}: ${nodeType} - ${nodeTitle} (${modeText})`);

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

                if (nodeDetails.length === 0) {
                    this.resultWidget.value = "No other nodes found in graph";
                    return;
                }

                const groups = graph._groups || [];
                let groupDetails = [];
                if (groups.length > 0) {
                    for (const group of groups) {
                        if (group.recomputeInsideNodes) {
                            group.recomputeInsideNodes();
                        }
                        const nodeCount = (group._nodes || []).length;
                        groupDetails.push(`@${group.title} (${nodeCount} nodes)`);
                    }
                }

                let result = `Found ${nodeDetails.length} nodes in graph:\n\n${nodeDetails.join('\n')}`;
                if (groupDetails.length > 0) {
                    result += `\n\n--- Groups (use @GroupName to bypass) ---\n${groupDetails.join('\n')}`;
                }
                this.resultWidget.value = result;

            } catch (error) {
                console.error('[NodeBypasser] Error listing nodes:', error);
                this.resultWidget.value = `Error: ${error.message}`;
            }
        }

        bypassNodesByName(bypass) {
            try {
                const nodeNamesText = this.nodeNamesInput.value;

                if (!nodeNamesText || nodeNamesText.trim() === "" || nodeNamesText.includes("Enter node names")) {
                    this.resultWidget.value = "Please enter node names to bypass";
                    return;
                }

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

                const nodeNames = nodeNamesText.split(',').map(name => name.trim()).filter(name => name.length > 0);
                const mainResults = this.processNodeList(nodeNames, nodes, bypass, "");
                processedCount += mainResults.processedCount;
                results.push(...mainResults.results);
                notFound.push(...mainResults.notFound);
                errors.push(...mainResults.errors);

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

        processNodeList(nodeNames, nodes, bypass, label = "") {
            const processedCount = { count: 0 };
            const results = [];
            const notFound = [];
            const errors = [];

            for (const nodeName of nodeNames) {
                try {
                    let matchingNodes = [];

                    if (nodeName.startsWith('@')) {
                        const groupName = nodeName.substring(1);
                        matchingNodes = this.findNodesInGroup(groupName);
                    }
                    else if (this.isRegexPattern(nodeName)) {
                        matchingNodes = this.findNodesByRegex(nodeName, nodes);
                    } else {
                        matchingNodes = nodes.filter(node =>
                            node !== this &&
                            (node.type.toLowerCase().includes(nodeName.toLowerCase()) ||
                             (node.title && node.title.toLowerCase().includes(nodeName.toLowerCase())) ||
                             (node.properties && node.properties.name && node.properties.name.toLowerCase().includes(nodeName.toLowerCase())) ||
                             (node.properties && node.properties.variable_name && node.properties.variable_name.toLowerCase().includes(nodeName.toLowerCase())) ||
                             (node.properties && node.properties.custom_name && node.properties.custom_name.toLowerCase().includes(nodeName.toLowerCase())) ||
                             (node._stableCustomName && node._stableCustomName.toLowerCase().includes(nodeName.toLowerCase())))
                        );
                    }

                    if (matchingNodes.length === 0) {
                        notFound.push(`${label}: ${nodeName}`);
                        continue;
                    }

                    for (const targetNode of matchingNodes) {
                        const newMode = bypass ? MODE_BYPASS : MODE_ALWAYS;
                        targetNode.mode = newMode;
                        processedCount.count++;
                        results.push(`${label ? '[' + label + '] ' : ''}Node ${targetNode.id}: ${targetNode.type}`);

                        if (targetNode.subgraph && targetNode.subgraph._nodes) {
                            for (const internalNode of targetNode.subgraph._nodes) {
                                internalNode.mode = newMode;
                                processedCount.count++;
                                results.push(`  └─ Internal: ${internalNode.id} (${internalNode.type})`);
                            }
                        }
                    }
                } catch (error) {
                    console.error(`[NodeBypasser] Error processing pattern "${nodeName}":`, error);
                    errors.push(`${label}: ${nodeName}: ${error.message}`);
                }
            }

            return { processedCount: processedCount.count, results, notFound, errors };
        }

        isRegexPattern(pattern) {
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

        findNodesByRegex(pattern, nodes) {
            let regexPattern = pattern;
            let isExclusion = false;

            if (pattern.startsWith('!')) {
                isExclusion = true;
                regexPattern = pattern.substring(1);
            }

            regexPattern = regexPattern
                .replace(/\*/g, '.*')
                .replace(/\?/g, '.');

            const regex = new RegExp(regexPattern, 'i');

            const matchingNodes = nodes.filter(node => {
                if (node === this) return false;

                const typeMatch = regex.test(node.type);
                const titleMatch = node.title ? regex.test(node.title) : false;
                const customNameMatch = node.properties && node.properties.custom_name ? regex.test(node.properties.custom_name) : false;
                const stableCustomNameMatch = node._stableCustomName ? regex.test(node._stableCustomName) : false;
                const matches = typeMatch || titleMatch || customNameMatch || stableCustomNameMatch;

                return isExclusion ? !matches : matches;
            });

            return matchingNodes;
        }

        findNodesInGroup(groupName) {
            const graph = app.graph;
            const groups = graph._groups || [];

            if (groups.length === 0) {
                return [];
            }

            let matchingGroups = [];
            const isWildcard = groupName.includes('*') || groupName.includes('?');

            if (isWildcard) {
                let regexPattern = groupName
                    .replace(/\*/g, '.*')
                    .replace(/\?/g, '.');
                const regex = new RegExp(`^${regexPattern}$`, 'i');
                matchingGroups = groups.filter(g => g.title && regex.test(g.title));
            } else {
                matchingGroups = groups.filter(g =>
                    g.title && g.title.toLowerCase().includes(groupName.toLowerCase())
                );
            }

            if (matchingGroups.length === 0) {
                return [];
            }

            const nodesInGroups = [];
            const seenNodeIds = new Set();

            for (const group of matchingGroups) {
                if (group.recomputeInsideNodes) {
                    group.recomputeInsideNodes();
                }

                const groupNodes = group._nodes || [];

                for (const node of groupNodes) {
                    if (node !== this && !seenNodeIds.has(node.id)) {
                        nodesInGroups.push(node);
                        seenNodeIds.add(node.id);
                    }
                }
            }

            return nodesInGroups;
        }
    }

    // Set up the node properties
    _NodeBypasser.type = "NodeBypasser";
    _NodeBypasser.title = "Node Bypasser";
    _NodeBypasser.category = "KNF_Utils";
    _NodeBypasser.description = "Bypass nodes by name";

    NodeBypasser = _NodeBypasser;
}

// Register the extension
app.registerExtension({
    name: "NV_Comfy_Utils.NodeBypasser",
    registerCustomNodes() {
        // Define the class here — LGraphNode is guaranteed to exist at this point
        defineNodeBypasser();
        LiteGraph.registerNodeType(NodeBypasser.type, NodeBypasser);
    },
    loadedGraphNode(node) {
        if (node.type == "NodeBypasser") {
            node._tempWidth = node.size[0];
        }
    },
});
