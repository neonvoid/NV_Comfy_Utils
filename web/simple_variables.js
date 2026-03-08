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
        // Global Canvas Patches for Hiding Managed Nodes
        // =====================================================

        function isManagedSetter(node) {
            return node?.type === "SetVariableNode" && node?.properties?._nv_managed === true;
        }

        // Patch 1: Suppress node rendering
        if (!LGraphCanvas.prototype._origDrawNode_nv) {
            LGraphCanvas.prototype._origDrawNode_nv = LGraphCanvas.prototype.drawNode;
            LGraphCanvas.prototype.drawNode = function(node, ctx) {
                if (isManagedSetter(node)) return;
                return this._origDrawNode_nv(node, ctx);
            };
        }

        // Patch 2: Suppress connection lines.
        // _renderAllLinkSegments is the low-level bottleneck for drawing bezier curves.
        // If it doesn't exist in this LiteGraph version, fall back to patching drawConnections.
        if (LGraphCanvas.prototype._renderAllLinkSegments) {
            if (!LGraphCanvas.prototype._origRenderAllLinkSegments_nv) {
                LGraphCanvas.prototype._origRenderAllLinkSegments_nv = LGraphCanvas.prototype._renderAllLinkSegments;
                LGraphCanvas.prototype._renderAllLinkSegments = function(ctx, link, startPos, endPos, ...rest) {
                    if (link && this.graph) {
                        // Cache the hidden state on the link object to avoid getNodeById every frame.
                        // Invalidate cache if endpoints changed (e.g. after link reconnection).
                        if (link._nv_last_origin !== link.origin_id || link._nv_last_target !== link.target_id) {
                            const originNode = this.graph.getNodeById(link.origin_id);
                            const targetNode = this.graph.getNodeById(link.target_id);
                            link._nv_hidden = isManagedSetter(originNode) || isManagedSetter(targetNode);
                            link._nv_last_origin = link.origin_id;
                            link._nv_last_target = link.target_id;
                        }
                        if (link._nv_hidden) return;
                    }
                    return this._origRenderAllLinkSegments_nv(ctx, link, startPos, endPos, ...rest);
                };
            }
        } else {
            // Fallback for LiteGraph versions that don't have _renderAllLinkSegments.
            // Patch drawConnections to temporarily skip links touching managed setters.
            if (!LGraphCanvas.prototype._origDrawConnections_nv) {
                LGraphCanvas.prototype._origDrawConnections_nv = LGraphCanvas.prototype.drawConnections;
                LGraphCanvas.prototype.drawConnections = function(ctx) {
                    // Build a set of managed-setter node ids for fast O(1) lookup.
                    const hiddenIds = new Set();
                    if (this.graph) {
                        for (const n of this.graph._nodes) {
                            if (isManagedSetter(n)) hiddenIds.add(n.id);
                        }
                    }
                    this._nv_hiddenLinkIds = hiddenIds;
                    const result = this._origDrawConnections_nv(ctx);
                    this._nv_hiddenLinkIds = null;
                    return result;
                };
            }
        }

        // Patch 3: Exclude managed setters from Ctrl+A (select-all).
        // LGraphCanvas.selectItems() calls this.graph.positionableItems() which is a
        // generator that yields ALL nodes including hidden setters. We wrap it so
        // managed setters are filtered before any selection code sees them.
        if (LGraph && LGraph.prototype.positionableItems && !LGraph.prototype._origPositionableItems_nv) {
            LGraph.prototype._origPositionableItems_nv = LGraph.prototype.positionableItems;
            LGraph.prototype.positionableItems = function* () {
                for (const item of this._origPositionableItems_nv()) {
                    if (!isManagedSetter(item)) yield item;
                }
            };
        }

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

                // Completely disable hit-testing — SetVariableNode is always managed,
                // so always return false regardless of coordinates.
                this.isPointInside = function(_x, _y, _margin) {
                    return false;
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

                // Update output type based on setter (explicit type > source output type)
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

                    // Follow the setter's input link to the source node's output type.
                    // The setter's own input type stays "*" (wildcard), so we must look
                    // at the actual source node's output slot for the real type.
                    const graph = node.graph;
                    const setterInput = setter.inputs?.[0];
                    if (graph && setterInput && setterInput.link != null) {
                        const link = graph._links?.get(setterInput.link)
                            ?? graph.links?.[setterInput.link];
                        if (link) {
                            const sourceNode = graph.getNodeById(link.origin_id);
                            const sourceType = sourceNode?.outputs?.[link.origin_slot]?.type;
                            if (sourceType && sourceType !== "*") {
                                this.outputs[0].type = sourceType;
                                this.outputs[0].name = sourceType;
                                return;
                            }
                        }
                    }

                    this.outputs[0].type = "*";
                    this.outputs[0].name = "*";
                };

                // Override getInputLink to pass through the setter's input link.
                // The execution engine calls getInputLink(outputSlotIndex) on virtual nodes
                // asking "what link feeds output slot N?". We always resolve through the
                // setter's single input (index 0), regardless of which output slot is queried.
                // NOTE: graph._links is a Map in ComfyUI frontend >= 1.10 — use .get().
                this.getInputLink = function(_slot) {
                    const setter = this.findSetter();
                    if (!setter) {
                        console.warn("[GetVariableNode] No setter found for:", this.widgets[0].value);
                        return null;
                    }
                    // Always use setter.inputs[0] — the setter has exactly one input.
                    const setterInput = setter.inputs[0];
                    if (!setterInput || setterInput.link == null) {
                        return null;
                    }
                    const graph = node.graph;
                    if (!graph) return null;
                    // graph._links is a Map; fall back to graph.links for older LiteGraph versions.
                    const link = graph._links?.get(setterInput.link)
                        ?? graph.links?.[setterInput.link]
                        ?? null;
                    return link;
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
