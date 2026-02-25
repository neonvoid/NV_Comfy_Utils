import { app } from "../../scripts/app.js";

// Clone With Connections — Alt+Shift+Drag to clone a node while preserving input connections.
// Normal Alt+Drag behavior is unchanged (clone without connections).
//
// How it works:
//   The ComfyUI Vue frontend handles Alt+Drag cloning via LGraphCanvas.cloneNodes() →
//   _deserializeItems() → graph.add(). We use a capture-phase pointerdown listener to
//   detect Alt+Shift, capture the original node's input connections, then temporarily
//   hook graph.add so that when the clone is added, we re-establish those connections.

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

// Convert pointer event to canvas-space coordinates
function eventToCanvasPos(e, canvas) {
    // Method 1: Use graph_mouse (set by processMouseMove, very close to click position)
    if (canvas.graph_mouse) {
        return canvas.graph_mouse;
    }
    // Method 2: Manual conversion via DragAndScale
    const ds = canvas.ds;
    const canvasEl = canvas.canvas;
    if (ds && canvasEl) {
        const rect = canvasEl.getBoundingClientRect();
        const x = (e.clientX - rect.left - ds.offset[0]) / ds.scale;
        const y = (e.clientY - rect.top - ds.offset[1]) / ds.scale;
        return [x, y];
    }
    return null;
}

app.registerExtension({
    name: "NV_Comfy_Utils.CloneWithConnections",

    setup() {
        // Track shift key state independently (pointerdown event doesn't always
        // reflect shiftKey reliably across all browsers/compositors)
        let shiftHeld = false;
        document.addEventListener("keydown", (e) => { if (e.key === "Shift") shiftHeld = true; }, true);
        document.addEventListener("keyup", (e) => { if (e.key === "Shift") shiftHeld = false; }, true);
        window.addEventListener("blur", () => { shiftHeld = false; });

        // Capture-phase pointerdown fires BEFORE the Vue handler that triggers cloneNodes.
        // We use it to detect Alt+Shift, find the node, and hook graph.add.
        document.addEventListener("pointerdown", function (e) {
            if (!e.altKey || !shiftHeld || e.ctrlKey) return;

            const canvas = app.canvas;
            const graph = app.graph;
            if (!canvas || !graph) return;

            // Find the node under the cursor
            const pos = eventToCanvasPos(e, canvas);
            if (!pos) return;

            const node = graph.getNodeOnPos(pos[0], pos[1]);
            if (!node) return;

            const connections = captureInputConnections(node, graph);
            if (connections.length === 0) return;

            console.log(
                LOG_PREFIX,
                `Captured ${connections.length} input connections from "${node.type}" (id=${node.id})`
            );

            // Hook graph.add — the clone path (cloneNodes → _deserializeItems) calls
            // graph.add(node) synchronously for each deserialized node.
            const origAdd = graph.add;
            const nodeType = node.type;

            const hookFn = function (newNode, ...args) {
                const result = origAdd.call(graph, newNode, ...args);

                // Only restore connections if this is actually the clone (matching type)
                if (newNode.type === nodeType) {
                    console.log(LOG_PREFIX, "Clone detected, restoring input connections...");
                    restoreInputConnections(newNode, connections, graph);
                    graph.setDirtyCanvas(true, true);
                }

                return result;
            };

            graph.add = hookFn;

            // Clean up after the current event loop tick — all synchronous graph.add
            // calls from _deserializeItems will have completed by then.
            setTimeout(() => {
                if (graph.add === hookFn) {
                    graph.add = origAdd;
                }
            }, 0);
        }, true); // capture phase — fires before Vue component handlers

        console.log(LOG_PREFIX, "Alt+Shift+Drag clone-with-connections ready");
    },
});
