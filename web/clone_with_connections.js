import { app } from "../../scripts/app.js";

// Clone With Connections — Alt+Shift+Drag to clone a node while preserving input connections.
// Normal Alt+Drag behavior is unchanged (clone without connections).

const LOG_PREFIX = "[CloneWithConnections]";

// Look up a link object, handling both Map (_links) and plain-object (links) access
function getLink(graph, linkId) {
    if (graph._links && typeof graph._links.get === "function") {
        return graph._links.get(linkId);
    }
    if (graph.links) {
        return graph.links[linkId];
    }
    return null;
}

// Capture input connections from a node: [{slot, origin_id, origin_slot}, ...]
function captureInputConnections(node, graph) {
    const connections = [];
    if (!node.inputs) return connections;

    for (let i = 0; i < node.inputs.length; i++) {
        const input = node.inputs[i];
        if (input.link != null) {
            const link = getLink(graph, input.link);
            if (link) {
                connections.push({
                    slot: i,
                    origin_id: link.origin_id,
                    origin_slot: link.origin_slot,
                });
            }
        }
    }
    return connections;
}

// Re-establish input connections on a cloned node
function restoreInputConnections(clone, connections, graph) {
    let restored = 0;
    for (const conn of connections) {
        const sourceNode = graph.getNodeById(conn.origin_id);
        if (sourceNode) {
            try {
                sourceNode.connect(conn.origin_slot, clone, conn.slot);
                restored++;
            } catch (err) {
                console.warn(LOG_PREFIX, "Failed to connect slot", conn.slot, err);
            }
        } else {
            console.warn(LOG_PREFIX, "Source node not found:", conn.origin_id);
        }
    }
    console.log(LOG_PREFIX, `Restored ${restored}/${connections.length} input connections`);
}

app.registerExtension({
    name: "NV_Comfy_Utils.CloneWithConnections",

    setup() {
        // State for the pending connection copy (set on Alt+Shift mousedown, consumed on graph.add)
        let pending = null;

        const origProcessMouseDown = LGraphCanvas.prototype.processMouseDown;
        LGraphCanvas.prototype.processMouseDown = function (e) {
            // Clean up any leftover hook from a previous Alt+Shift click that never dragged
            if (pending) {
                this.graph.add = pending.origAdd;
                pending = null;
            }

            // Alt+Shift+Click (no Ctrl) on a node → prepare to copy connections after clone
            if (
                e.altKey &&
                e.shiftKey &&
                !e.ctrlKey &&
                LiteGraph.alt_drag_do_clone_nodes &&
                this.allow_interaction
            ) {
                // Convert screen coords to canvas space
                let canvasPos;
                if (typeof this.convertEventToCanvasOffset === "function") {
                    canvasPos = this.convertEventToCanvasOffset(e);
                } else if (this.ds && typeof this.ds.convertOffsetToCanvas === "function") {
                    canvasPos = this.ds.convertOffsetToCanvas(e.offsetX, e.offsetY);
                } else {
                    canvasPos = this.graph_mouse;
                }

                if (canvasPos) {
                    const node = this.graph.getNodeOnPos(canvasPos[0], canvasPos[1]);

                    if (node) {
                        const connections = captureInputConnections(node, this.graph);

                        if (connections.length > 0) {
                            const graph = this.graph;
                            const origAdd = graph.add;

                            pending = { origAdd };

                            // Temporarily hook graph.add to intercept the clone being added
                            graph.add = function (newNode, skipComputeOrder) {
                                // Restore immediately
                                graph.add = pending.origAdd;
                                pending = null;

                                // Call the real graph.add
                                const result = origAdd.call(graph, newNode, skipComputeOrder);

                                // Copy input connections to the clone
                                console.log(LOG_PREFIX, "Clone added, restoring input connections...");
                                restoreInputConnections(newNode, connections, graph);

                                // Force visual redraw of new links (gotcha #22)
                                graph.setDirtyCanvas(true, true);

                                return result;
                            };

                            console.log(
                                LOG_PREFIX,
                                `Captured ${connections.length} input connections from "${node.type}" (id=${node.id})`
                            );
                        }
                    }
                }
            }

            // Call original processMouseDown (native alt-clone runs inside)
            return origProcessMouseDown.call(this, e);
        };

        console.log(LOG_PREFIX, "Alt+Shift+Drag clone-with-connections ready");
    },
});
