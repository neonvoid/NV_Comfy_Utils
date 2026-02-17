/**
 * Simple Variables — SetVariableNode & GetVariableNode
 *
 * SetVariableNode is now hidden/managed by the Variables Panel console.
 * Users never create or interact with SetVariableNodes directly.
 * The VariableManager handles all creation and lifecycle.
 *
 * GetVariableNode remains user-visible and is placed via drag-and-drop
 * from the panel or programmatically by the VariableManager.
 */

import { app } from "../../scripts/app.js";
import { variableManager } from "./variable_manager.js";

console.log("[NV_Comfy_Utils] Loading simple variables...");

app.registerExtension({
    name: "NV_Comfy_Utils.SimpleVariables",

    registerCustomNodes() {
        console.log("[NV_Comfy_Utils] Registering simple variable nodes...");

        // =====================================================
        // SetVariableNode — Hidden, auto-managed by console
        // =====================================================
        class SetVariableNode extends LGraphNode {
            constructor(title) {
                super(title);

                this.defaultVisibility = true;
                this.serialize_widgets = true;
                this.drawConnection = false;
                this.slotColor = "#FFF";
                this.canvas = app.canvas;

                if (!this.properties) {
                    this.properties = {};
                }
                this.properties.previousName = "";
                this.properties._nv_managed = true;
                this.properties.sourceNodeId = null;
                this.properties.sourceSlotIndex = null;

                const node = this;

                // Variable name widget (managed programmatically, not by user)
                this.addWidget(
                    "text",
                    "Variable Name",
                    "",
                    (value) => {
                        this.properties.previousName = value;
                        this.updateGetters();
                    }
                );

                // Input and output
                this.addInput("*", "*");
                this.addOutput("*", '*');

                // Connection change handler
                this.onConnectionsChange = function(slotType, slot, isChangeConnect, link_info, output) {
                    if (slotType === 0) { // Input connection changed
                        this.updateGetters();
                    }
                };

                // Update all getter nodes that use this variable name
                this.updateGetters = function() {
                    if (!node.graph) return;

                    const getters = node.graph._nodes.filter(otherNode =>
                        otherNode.type === 'GetVariableNode' &&
                        otherNode.widgets[0].value === this.widgets[0].value &&
                        this.widgets[0].value !== ''
                    );

                    getters.forEach(getter => {
                        if (getter.updateType) {
                            getter.updateType();
                        }
                    });
                };

                // Find all getter nodes that reference this setter
                this.findGetters = function() {
                    if (!node.graph) return [];

                    const name = this.widgets[0].value;
                    if (!name || name === '') return [];

                    return node.graph._nodes.filter(otherNode =>
                        otherNode.type === 'GetVariableNode' &&
                        otherNode.widgets[0].value === name
                    );
                };

                // Helper to connect a source node output to this setter's input
                this.connectToSource = function(sourceNode, slotIndex) {
                    if (this.inputs[0] && this.inputs[0].link != null) {
                        node.graph.removeLink(this.inputs[0].link);
                    }
                    sourceNode.connect(slotIndex, this, 0);
                    this.properties.sourceNodeId = sourceNode.id;
                    this.properties.sourceSlotIndex = slotIndex;
                };

                // This is a virtual node - frontend handles everything
                this.isVirtualNode = true;

                // ===== Hidden rendering for managed setters =====

                // Store original draw methods
                const _origDrawForeground = this.onDrawForeground;
                const _origDrawBackground = this.onDrawBackground;

                this.onDrawForeground = function(ctx) {
                    if (this.properties._nv_managed) return; // Don't render
                    if (_origDrawForeground) _origDrawForeground.call(this, ctx);
                };

                this.onDrawBackground = function(ctx) {
                    if (this.properties._nv_managed) return; // Don't render
                    if (_origDrawBackground) _origDrawBackground.call(this, ctx);
                };

                // Make managed setters invisible to box-select
                const _origIsPointInside = this.isPointInside;
                this.isPointInside = function(x, y, margin) {
                    if (this.properties._nv_managed) return false;
                    if (_origIsPointInside) return _origIsPointInside.call(this, x, y, margin);
                    return LGraphNode.prototype.isPointInside.call(this, x, y, margin);
                };

                // On configure (deserialization), ensure managed mode
                this.onConfigure = function(info) {
                    if (!this.properties) this.properties = {};
                    // Auto-migrate old manually-placed setters
                    if (!this.properties._nv_managed) {
                        this.properties._nv_managed = true;
                    }
                };
            }
        }

        // =====================================================
        // GetVariableNode — User-visible, placed via drag-and-drop
        // =====================================================
        class GetVariableNode extends LGraphNode {
            constructor(title) {
                super(title);

                this.defaultVisibility = true;
                this.serialize_widgets = true;
                this.drawConnection = false;
                this.slotColor = "#FFF";
                this.canvas = app.canvas;

                if (!this.properties) {
                    this.properties = {};
                }

                const node = this;

                // Combo widget — values pulled from VariableManager
                this.addWidget(
                    "combo",
                    "Variable Name",
                    "",
                    (value) => {
                        this.updateType();
                    },
                    {
                        values: () => {
                            return variableManager.getVariableNames();
                        }
                    }
                );

                // Output
                this.addOutput("*", '*');

                // Find the corresponding setter node
                this.findSetter = function() {
                    const name = this.widgets[0].value;
                    if (!name || !node.graph) return null;

                    return node.graph._nodes.find(otherNode =>
                        otherNode.type === 'SetVariableNode' &&
                        otherNode.widgets[0].value === name
                    );
                };

                // Update output type based on setter (explicit type > connection-inferred)
                this.updateType = function() {
                    const setter = this.findSetter();
                    if (!setter) {
                        this.outputs[0].type = "*";
                        this.outputs[0].name = "*";
                        return;
                    }

                    // Explicit type set by user in panel takes priority
                    const explicit = setter.properties?.explicitType;
                    if (explicit && explicit !== "*") {
                        this.outputs[0].type = explicit;
                        this.outputs[0].name = explicit;
                        return;
                    }

                    // Fall back to connection-inferred type
                    if (setter.inputs[0].type && setter.inputs[0].type !== "*") {
                        this.outputs[0].type = setter.inputs[0].type;
                        this.outputs[0].name = setter.inputs[0].type;
                    } else {
                        this.outputs[0].type = "*";
                        this.outputs[0].name = "*";
                    }
                };

                // Override getInputLink to pass through the setter's input link
                this.getInputLink = function(slot) {
                    const setter = this.findSetter();
                    if (setter && setter.inputs[slot]) {
                        const slotInfo = setter.inputs[slot];
                        const link = node.graph.links[slotInfo.link];
                        return link;
                    } else {
                        console.warn("[GetVariableNode] No setter found for:", this.widgets[0].value);
                    }
                    return null;
                };

                // Navigate to the setter node (still useful for debugging)
                this.goToSetter = function() {
                    const setter = this.findSetter();
                    if (setter) {
                        this.canvas.centerOnNode(setter);
                        this.canvas.selectNode(setter, false);
                        this.canvas.setDirty(true, true);
                    }
                };

                this.isVirtualNode = true;

                this.onAdded = function(graph) {
                    // Update type when added to graph
                    this.updateType();
                };
            }

            // Context menu for GetVariableNode
            getExtraMenuOptions(_, options) {
                const varName = this.widgets?.[0]?.value;
                if (varName) {
                    const sourceInfo = variableManager.getSourceInfo(varName);
                    if (sourceInfo) {
                        options.unshift({
                            content: `Go to source: ${sourceInfo.nodeTitle}`,
                            callback: () => {
                                app.canvas.centerOnNode(sourceInfo.node);
                                app.canvas.selectNode(sourceInfo.node, false);
                                app.canvas.setDirty(true, true);
                            },
                        });
                    }
                }
            }
        }

        // Register both node types (required for graph serialization)
        LiteGraph.registerNodeType("SetVariableNode", SetVariableNode);
        LiteGraph.registerNodeType("GetVariableNode", GetVariableNode);

        console.log("[NV_Comfy_Utils] Simple variable nodes registered");
    },

    async setup() {
        // After all nodes are loaded, hide SetVariableNode from menus
        // and migrate any existing manually-placed setters
        variableManager.hideSetterFromMenus();

        if (app.graph) {
            variableManager.migrateExistingSetters();
        } else {
            // Wait for graph to be ready, then migrate
            const check = setInterval(() => {
                if (app.graph) {
                    clearInterval(check);
                    variableManager.migrateExistingSetters();
                }
            }, 100);
        }
    }
});
