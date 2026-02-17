/**
 * Variable Manager — Central API for console-driven variable CRUD
 *
 * All variable creation, deletion, source assignment, and getter placement
 * flows through this singleton. Both the Variables Panel and the
 * "Promote to Variable" context menu delegate to it.
 *
 * Under the hood, SetVariableNodes still exist in the graph (required for
 * getInputLink() interception in GetVariableNode), but they are hidden
 * and auto-managed. Users never see or interact with them directly.
 */

import { app } from "../../scripts/app.js";

// Type color map inspired by UE Blueprint color coding
const TYPE_COLORS = {
    IMAGE: "#5b8def",
    LATENT: "#a855f7",
    MODEL: "#4ade80",
    CONDITIONING: "#f59e0b",
    CLIP: "#f472b6",
    VAE: "#e879f9",
    MASK: "#94a3b8",
    STRING: "#ec4899",
    INT: "#2dd4bf",
    FLOAT: "#facc15",
    BOOLEAN: "#f87171",
    CONTROL_NET: "#818cf8",
    "*": "#888",
    unconnected: "#666",
};

// Core ComfyUI types (always shown at top of type list)
const CORE_TYPES = [
    "*",
    "IMAGE",
    "LATENT",
    "CONDITIONING",
    "MODEL",
    "CLIP",
    "VAE",
    "MASK",
    "STRING",
    "INT",
    "FLOAT",
    "BOOLEAN",
    "CONTROL_NET",
    "CLIP_VISION",
    "CLIP_VISION_OUTPUT",
    "STYLE_MODEL",
    "GLIGEN",
    "UPSCALE_MODEL",
    "SIGMAS",
    "NOISE",
    "SAMPLER",
    "GUIDER",
];

/**
 * Dynamically discover all slot types registered in the current LiteGraph session.
 * This picks up types from custom nodes (Impact Pack, KJNodes, etc.) automatically.
 * Returns core types first, then any additional discovered types sorted alphabetically.
 */
function discoverAllTypes() {
    const allTypes = new Set(CORE_TYPES);

    // Discover from LiteGraph slot_types_default_in (input slot types)
    if (LiteGraph.slot_types_default_in) {
        for (const typeName of Object.keys(LiteGraph.slot_types_default_in)) {
            if (typeName && typeName !== "" && typeName !== "undefined") {
                allTypes.add(typeName);
            }
        }
    }

    // Discover from LiteGraph slot_types_default_out (output slot types)
    if (LiteGraph.slot_types_default_out) {
        for (const typeName of Object.keys(LiteGraph.slot_types_default_out)) {
            if (typeName && typeName !== "" && typeName !== "undefined") {
                allTypes.add(typeName);
            }
        }
    }

    // Discover from registered node definitions (scan all input/output slot types)
    if (LiteGraph.registered_node_types) {
        for (const nodeType of Object.values(LiteGraph.registered_node_types)) {
            try {
                // Check if the node class has default inputs/outputs defined
                const proto = nodeType.prototype;
                if (proto && proto.inputs) {
                    for (const input of proto.inputs) {
                        if (input && input.type && input.type !== "*") {
                            allTypes.add(input.type);
                        }
                    }
                }
                if (proto && proto.outputs) {
                    for (const output of proto.outputs) {
                        if (output && output.type && output.type !== "*") {
                            allTypes.add(output.type);
                        }
                    }
                }
            } catch (e) {
                // Skip nodes that error during introspection
            }
        }
    }

    // Build final list: core types first (in order), then extras sorted
    const coreSet = new Set(CORE_TYPES);
    const extras = [...allTypes].filter(t => !coreSet.has(t)).sort();

    return [...CORE_TYPES, ...extras];
}

const MANAGED_SETTER_X = -5000;
const MANAGED_SETTER_Y_SPACING = 60;

class VariableManager {
    constructor() {
        this._setterClass = null; // Stored after registration
    }

    // ===== CRUD =====

    /**
     * Create a new variable. Places a hidden SetVariableNode offscreen.
     * @param {string} name - Variable name
     * @param {string} [explicitType="*"] - Explicit data type (e.g. "IMAGE", "LATENT")
     * Returns the created setter node, or null if name already exists.
     */
    createVariable(name, explicitType = "*") {
        if (!name || !app.graph) return null;

        // Check for duplicate name
        const existing = this._findSetter(name);
        if (existing) {
            console.warn(`[VariableManager] Variable "${name}" already exists`);
            return null;
        }

        // Create the SetVariableNode programmatically
        const setter = LiteGraph.createNode("SetVariableNode");
        if (!setter) {
            console.error("[VariableManager] Failed to create SetVariableNode");
            return null;
        }

        // Configure as managed
        setter.properties._nv_managed = true;
        setter.properties.explicitType = explicitType;
        setter.widgets[0].value = name;
        setter.properties.previousName = name;
        setter.title = `_var_${name}`;

        // Set the input/output slot types to match the explicit type
        if (explicitType && explicitType !== "*") {
            setter.inputs[0].type = explicitType;
            setter.outputs[0].type = explicitType;
        }

        // Position offscreen
        const index = this._countManagedSetters();
        setter.pos = [MANAGED_SETTER_X, index * MANAGED_SETTER_Y_SPACING];

        app.graph.add(setter);
        app.graph.setDirtyCanvas(true, false);

        console.log(`[VariableManager] Created variable "${name}" (type: ${explicitType})`);
        return setter;
    }

    /**
     * Delete a variable and ALL associated nodes (setter + all getters).
     */
    deleteVariable(name) {
        if (!name || !app.graph) return;

        const nodes = app.graph._nodes || [];

        // Find all getters for this variable
        const getters = nodes.filter(n =>
            n.type === "GetVariableNode" &&
            n.widgets?.[0]?.value === name
        );

        // Find all setters for this variable (should be 1, but handle duplicates)
        const setters = nodes.filter(n =>
            n.type === "SetVariableNode" &&
            n.widgets?.[0]?.value === name
        );

        // Remove getters first
        for (const getter of getters) {
            app.graph.remove(getter);
        }

        // Remove setters
        for (const setter of setters) {
            app.graph.remove(setter);
        }

        app.graph.setDirtyCanvas(true, false);
        console.log(`[VariableManager] Deleted variable "${name}" (${setters.length} setter(s), ${getters.length} getter(s))`);
    }

    /**
     * Rename a variable across all setter and getter nodes.
     */
    renameVariable(oldName, newName) {
        if (!oldName || !newName || oldName === newName || !app.graph) return false;

        // Check for collision
        const existingTarget = this._findSetter(newName);
        if (existingTarget) {
            console.warn(`[VariableManager] Cannot rename: "${newName}" already exists`);
            return false;
        }

        const nodes = app.graph._nodes || [];

        // Update all setters
        const setters = nodes.filter(n =>
            n.type === "SetVariableNode" &&
            n.widgets?.[0]?.value === oldName
        );
        for (const setter of setters) {
            setter.widgets[0].value = newName;
            setter.properties.previousName = newName;
            setter.title = `_var_${newName}`;
        }

        // Update all getters
        const getters = nodes.filter(n =>
            n.type === "GetVariableNode" &&
            n.widgets?.[0]?.value === oldName
        );
        for (const getter of getters) {
            getter.widgets[0].value = newName;
        }

        // Trigger type propagation
        for (const setter of setters) {
            if (setter.updateGetters) {
                setter.updateGetters();
            }
        }

        app.graph.setDirtyCanvas(true, false);
        console.log(`[VariableManager] Renamed "${oldName}" → "${newName}"`);
        return true;
    }

    // ===== Source Assignment =====

    /**
     * Assign a data source to a variable (the "Promote to Variable" action).
     * Links sourceNode's output slot to the hidden setter's input.
     */
    assignSource(varName, sourceNode, slotIndex) {
        if (!app.graph) return false;

        let setter = this._findSetter(varName);
        if (!setter) {
            console.warn(`[VariableManager] Variable "${varName}" not found`);
            return false;
        }

        // Remove existing input link on the setter
        if (setter.inputs[0] && setter.inputs[0].link != null) {
            app.graph.removeLink(setter.inputs[0].link);
        }

        // Create new link from source output to setter input
        sourceNode.connect(slotIndex, setter, 0);

        // Store source reference in properties
        setter.properties.sourceNodeId = sourceNode.id;
        setter.properties.sourceSlotIndex = slotIndex;

        // Update all getters to reflect the new type
        if (setter.updateGetters) {
            setter.updateGetters();
        }

        app.graph.setDirtyCanvas(true, false);
        console.log(`[VariableManager] Assigned source for "${varName}": node ${sourceNode.id} slot ${slotIndex}`);
        return true;
    }

    /**
     * Set the explicit data type for a variable.
     * Updates the setter's slot types and propagates to all getters.
     */
    setVariableType(varName, newType) {
        if (!app.graph) return false;

        const setter = this._findSetter(varName);
        if (!setter) return false;

        setter.properties.explicitType = newType;

        // Update setter slot types
        if (newType && newType !== "*") {
            setter.inputs[0].type = newType;
            setter.outputs[0].type = newType;
        } else {
            // Revert to wildcard or connection-inferred type
            setter.inputs[0].type = "*";
            setter.outputs[0].type = "*";
        }

        // Propagate to all getters
        if (setter.updateGetters) {
            setter.updateGetters();
        }

        app.graph.setDirtyCanvas(true, false);
        console.log(`[VariableManager] Set type for "${varName}" → ${newType}`);
        return true;
    }

    /**
     * Get the explicit type set for a variable, or null if none.
     */
    getExplicitType(varName) {
        const setter = this._findSetter(varName);
        if (!setter) return null;
        const t = setter.properties?.explicitType;
        return (t && t !== "*") ? t : null;
    }

    /**
     * Disconnect the source from a variable's setter.
     */
    unassignSource(varName) {
        if (!app.graph) return false;

        const setter = this._findSetter(varName);
        if (!setter) return false;

        if (setter.inputs[0] && setter.inputs[0].link != null) {
            app.graph.removeLink(setter.inputs[0].link);
        }

        setter.properties.sourceNodeId = null;
        setter.properties.sourceSlotIndex = null;

        if (setter.updateGetters) {
            setter.updateGetters();
        }

        app.graph.setDirtyCanvas(true, false);
        return true;
    }

    // ===== Getter Creation (drag-and-drop targets) =====

    /**
     * Create a GetVariableNode at a canvas position (drop on empty canvas).
     */
    createGetter(varName, canvasPos) {
        if (!app.graph) return null;

        const getter = LiteGraph.createNode("GetVariableNode");
        if (!getter) return null;

        getter.pos = [canvasPos[0], canvasPos[1]];
        getter.widgets[0].value = varName;
        app.graph.add(getter);

        // Update type from setter
        if (getter.updateType) {
            getter.updateType();
        }

        app.canvas.selectNode(getter, false);
        app.graph.setDirtyCanvas(true, false);

        console.log(`[VariableManager] Created getter for "${varName}" at [${Math.round(canvasPos[0])}, ${Math.round(canvasPos[1])}]`);
        return getter;
    }

    /**
     * Create a GetVariableNode and connect it to a target node's input slot
     * (drop on node input).
     */
    createGetterAndConnect(varName, targetNode, targetSlotIndex) {
        if (!app.graph) return null;

        // Position the getter to the left of the target node
        const offsetX = 220;
        const slotY = targetNode.pos[1] + (targetSlotIndex * 30) + 30;
        const canvasPos = [targetNode.pos[0] - offsetX, slotY];

        const getter = this.createGetter(varName, canvasPos);
        if (!getter) return null;

        // Connect getter output to target input
        getter.connect(0, targetNode, targetSlotIndex);

        app.graph.setDirtyCanvas(true, false);
        console.log(`[VariableManager] Connected getter "${varName}" → node ${targetNode.id} input ${targetSlotIndex}`);
        return getter;
    }

    // ===== Queries =====

    /**
     * Get info about a single variable.
     */
    getVariable(name) {
        if (!app.graph) return null;

        const nodes = app.graph._nodes || [];
        const setter = this._findSetter(name);
        const getters = nodes.filter(n =>
            n.type === "GetVariableNode" &&
            n.widgets?.[0]?.value === name
        );

        if (!setter && getters.length === 0) return null;

        return {
            name,
            setter,
            getters,
            type: setter ? this._resolveType(setter) : "*",
            isConnected: setter ? this._isSetterConnected(setter) : false,
        };
    }

    /**
     * Get all variables as a Map. Includes orphans and duplicates info.
     */
    getAllVariables() {
        if (!app.graph) return new Map();

        const nodes = app.graph._nodes || [];
        const variableMap = new Map();

        // Pass 1: Collect all setters
        const setters = nodes.filter(n => n.type === "SetVariableNode");
        for (const setter of setters) {
            const name = setter.widgets?.[0]?.value;
            if (!name || name === "") continue;

            if (!variableMap.has(name)) {
                variableMap.set(name, {
                    name,
                    type: this._resolveType(setter),
                    explicitType: setter.properties?.explicitType || "*",
                    isConnected: this._isSetterConnected(setter),
                    setter: setter,
                    getters: [],
                    hasOrphanGetters: false,
                    hasDuplicateSetters: false,
                    allSetters: [setter],
                    sourceNodeId: setter.properties?.sourceNodeId || null,
                    sourceSlotIndex: setter.properties?.sourceSlotIndex ?? null,
                });
            } else {
                const info = variableMap.get(name);
                info.hasDuplicateSetters = true;
                info.allSetters.push(setter);
            }
        }

        // Pass 2: Collect all getters, match to setters
        const getterNodes = nodes.filter(n => n.type === "GetVariableNode");
        for (const getter of getterNodes) {
            const name = getter.widgets?.[0]?.value;
            if (!name || name === "") continue;

            if (variableMap.has(name)) {
                variableMap.get(name).getters.push(getter);
            } else {
                variableMap.set(name, {
                    name,
                    type: "*",
                    isConnected: false,
                    setter: null,
                    getters: [getter],
                    hasOrphanGetters: true,
                    hasDuplicateSetters: false,
                    allSetters: [],
                    sourceNodeId: null,
                    sourceSlotIndex: null,
                });
            }
        }

        return variableMap;
    }

    /**
     * Get sorted list of variable names (for combo widgets and menus).
     */
    getVariableNames() {
        if (!app.graph) return [];

        const nodes = app.graph._nodes || [];
        const names = new Set();

        for (const n of nodes) {
            if (n.type === "SetVariableNode") {
                const name = n.widgets?.[0]?.value;
                if (name && name !== "") names.add(name);
            }
        }

        return [...names].sort();
    }

    /**
     * Get the color for a variable type.
     * Known types get curated colors; unknown types (from custom nodes)
     * get a deterministic color generated from the type name.
     */
    getTypeColor(type) {
        if (TYPE_COLORS[type]) return TYPE_COLORS[type];

        // Generate a deterministic HSL color from the type name string
        // so custom node types always get a consistent color
        let hash = 0;
        for (let i = 0; i < type.length; i++) {
            hash = type.charCodeAt(i) + ((hash << 5) - hash);
        }
        const hue = Math.abs(hash) % 360;
        return `hsl(${hue}, 55%, 60%)`;
    }

    /**
     * Get the list of available variable types for dropdowns.
     * Dynamically discovers types from all registered nodes (including custom nodes).
     * Core types appear first, then additional types sorted alphabetically.
     */
    getVariableTypes() {
        return discoverAllTypes();
    }

    /**
     * Compute a quick hash for dirty-checking (used by panel refresh loop).
     */
    computeQuickHash() {
        if (!app.graph) return "";
        const nodes = app.graph._nodes || [];
        let hash = "";
        for (const n of nodes) {
            if (n.type === "SetVariableNode" || n.type === "GetVariableNode") {
                hash += `${n.id}=${n.widgets?.[0]?.value}|${n.inputs?.[0]?.type}|${n.inputs?.[0]?.link}|${n.properties?.explicitType},`;
            }
        }
        return hash;
    }

    // ===== Canvas Helpers =====

    /**
     * Find which input slot (if any) is under a canvas position on a given node.
     * Returns the slot index or -1 if none.
     */
    findInputSlotAtPos(node, canvasX, canvasY) {
        if (!node.inputs) return -1;

        const threshold = 15; // px threshold for slot hit detection
        for (let i = 0; i < node.inputs.length; i++) {
            const slotPos = node.getConnectionPos(true, i);
            if (!slotPos) continue;

            const dx = canvasX - slotPos[0];
            const dy = canvasY - slotPos[1];
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < threshold) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Get the source node info for a variable (which node feeds the setter).
     */
    getSourceInfo(varName) {
        const setter = this._findSetter(varName);
        if (!setter || !setter.inputs[0] || setter.inputs[0].link == null) {
            return null;
        }

        const link = app.graph.links[setter.inputs[0].link];
        if (!link) return null;

        const sourceNode = app.graph.getNodeById(link.origin_id);
        if (!sourceNode) return null;

        const slotIndex = link.origin_slot;
        const outputName = sourceNode.outputs?.[slotIndex]?.name || `output_${slotIndex}`;

        return {
            node: sourceNode,
            slotIndex,
            outputName,
            nodeTitle: sourceNode.title || sourceNode.type,
        };
    }

    // ===== Backward Compatibility =====

    /**
     * Migrate any old manually-placed SetVariableNodes to managed mode.
     * Called on extension setup.
     */
    migrateExistingSetters() {
        if (!app.graph) return;

        const nodes = app.graph._nodes || [];
        const setters = nodes.filter(n =>
            n.type === "SetVariableNode" && !n.properties?._nv_managed
        );

        for (const setter of setters) {
            if (!setter.properties) setter.properties = {};
            setter.properties._nv_managed = true;
            setter.title = `_var_${setter.widgets?.[0]?.value || "unnamed"}`;

            // Reposition offscreen
            const index = this._countManagedSetters();
            setter.pos = [MANAGED_SETTER_X, index * MANAGED_SETTER_Y_SPACING];

            console.log(`[VariableManager] Migrated legacy SetVariableNode "${setter.widgets?.[0]?.value}" to managed mode`);
        }

        if (setters.length > 0) {
            app.graph.setDirtyCanvas(true, false);
        }
    }

    /**
     * Remove SetVariableNode from LiteGraph slot default arrays
     * so it doesn't appear in link-release menus.
     */
    hideSetterFromMenus() {
        const nodeTypeName = "SetVariableNode";

        // Remove from input slot defaults
        if (LiteGraph.slot_types_default_in) {
            for (const arr of Object.values(LiteGraph.slot_types_default_in)) {
                const idx = arr.indexOf(nodeTypeName);
                if (idx !== -1) arr.splice(idx, 1);
            }
        }

        // Remove from output slot defaults
        if (LiteGraph.slot_types_default_out) {
            for (const arr of Object.values(LiteGraph.slot_types_default_out)) {
                const idx = arr.indexOf(nodeTypeName);
                if (idx !== -1) arr.splice(idx, 1);
            }
        }
    }

    // ===== Private Helpers =====

    _findSetter(name) {
        if (!app.graph) return null;
        return (app.graph._nodes || []).find(n =>
            n.type === "SetVariableNode" &&
            n.widgets?.[0]?.value === name
        );
    }

    _findGetters(name) {
        if (!app.graph) return [];
        return (app.graph._nodes || []).filter(n =>
            n.type === "GetVariableNode" &&
            n.widgets?.[0]?.value === name
        );
    }

    _resolveType(setter) {
        // Explicit type takes priority
        const explicit = setter.properties?.explicitType;
        if (explicit && explicit !== "*") {
            return explicit;
        }
        // Fall back to connection-inferred type
        if (setter.inputs && setter.inputs[0]) {
            const inputType = setter.inputs[0].type;
            return (inputType && inputType !== "*") ? inputType : "unconnected";
        }
        return "unconnected";
    }

    _isSetterConnected(setter) {
        if (!setter.inputs || !setter.inputs[0]) return false;
        return setter.inputs[0].link != null;
    }

    _countManagedSetters() {
        if (!app.graph) return 0;
        return (app.graph._nodes || []).filter(n =>
            n.type === "SetVariableNode" &&
            n.properties?._nv_managed
        ).length;
    }
}

// Singleton export
export const variableManager = new VariableManager();

// Also expose on window for debugging / cross-module access
window.NVVariableManager = variableManager;

console.log("[NV_Comfy_Utils] Variable Manager loaded");
