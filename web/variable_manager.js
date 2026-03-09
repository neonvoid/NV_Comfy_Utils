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
        this._inUndo = false; // Reentrancy guard for nested _withUndo calls
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

        return this._withUndo(() => {
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
        });
    }

    /**
     * Delete a variable and ALL associated nodes (setter + all getters).
     */
    deleteVariable(name) {
        if (!name || !app.graph) return;

        this._withUndo(() => {
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
        });
    }

    /**
     * Rename a variable across all setter and getter nodes.
     */
    renameVariable(oldName, newName) {
        if (!oldName || !newName || oldName === newName || !app.graph) return false;

        return this._withUndo(() => {
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
        });
    }

    // ===== Source Assignment =====

    /**
     * Assign a data source to a variable (the "Promote to Variable" action).
     * Links sourceNode's output slot to the hidden setter's input.
     */
    assignSource(varName, sourceNode, slotIndex) {
        if (!app.graph) return false;

        return this._withUndo(() => {
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
        });
    }

    /**
     * Set the explicit data type for a variable.
     * Updates the setter's slot types and propagates to all getters.
     */
    setVariableType(varName, newType) {
        if (!app.graph) return false;

        return this._withUndo(() => {
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
        });
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

        return this._withUndo(() => {
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
        });
    }

    // ===== Source Pool =====

    /**
     * Add a source node to a variable's candidate pool.
     * @param {string} varName - Variable name
     * @param {number} nodeId - Source node ID
     * @param {number} slotIndex - Output slot index on the source node
     * @param {string} [label] - User-friendly label (defaults to "NodeTitle:slotName")
     * @returns {string|null} The candidate ID if added, null on failure
     */
    addToPool(varName, nodeId, slotIndex, label) {
        if (!app.graph) return null;

        return this._withUndo(() => {
            const setter = this._findSetter(varName);
            if (!setter) {
                console.warn(`[VariableManager] Variable "${varName}" not found`);
                return null;
            }

            const sourceNode = app.graph.getNodeById(nodeId);
            if (!sourceNode) {
                console.warn(`[VariableManager] Source node ${nodeId} not found`);
                return null;
            }

            // Type check: verify the source output is compatible
            const varType = this._resolveType(setter);
            const outputSlot = sourceNode.outputs?.[slotIndex];
            if (!outputSlot) {
                console.warn(`[VariableManager] Node ${nodeId} has no output slot ${slotIndex}`);
                return null;
            }
            if (this._isTypeIncompatible(varType, outputSlot.type)) {
                console.warn(`[VariableManager] Type mismatch: variable "${varName}" is ${varType}, source outputs ${outputSlot.type}`);
                return null;
            }

            // Initialize pool if needed
            if (!setter.properties.sourceCandidates) {
                setter.properties.sourceCandidates = [];
            }

            // Check for duplicate (same nodeId + slotIndex)
            const existing = setter.properties.sourceCandidates.find(
                c => c.nodeId === nodeId && c.slotIndex === slotIndex
            );
            if (existing) {
                console.log(`[VariableManager] Source already in pool for "${varName}"`);
                return existing.id;
            }

            // Generate a stable ID
            const id = `pool_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

            // Build label from node info if not provided
            const outputName = outputSlot.name || `output_${slotIndex}`;
            const nodeTitle = sourceNode.title || sourceNode.type;
            const finalLabel = label || `${nodeTitle}:${outputName}`;

            const candidate = {
                id,
                nodeId,
                nodeTitle,
                nodeType: sourceNode.type,
                slotIndex,
                outputType: outputSlot.type,
                label: finalLabel,
            };

            setter.properties.sourceCandidates.push(candidate);

            // If this is the first candidate and variable has no source, auto-activate it
            if (setter.properties.sourceCandidates.length === 1 && !this._isSetterConnected(setter)) {
                this._activateCandidate(setter, candidate);
            }

            app.graph.setDirtyCanvas(true, false);
            console.log(`[VariableManager] Added "${finalLabel}" to pool for "${varName}"`);
            return id;
        });
    }

    /**
     * Remove a candidate from a variable's source pool.
     * If the removed candidate was active, switches to the first remaining candidate.
     */
    removeFromPool(varName, candidateId) {
        if (!app.graph) return false;

        return this._withUndo(() => {
            const setter = this._findSetter(varName);
            if (!setter || !setter.properties.sourceCandidates) return false;

            const idx = setter.properties.sourceCandidates.findIndex(c => c.id === candidateId);
            if (idx === -1) return false;

            const wasActive = setter.properties.activeSourceId === candidateId;
            setter.properties.sourceCandidates.splice(idx, 1);

            // If the active candidate was removed, switch to first valid remaining
            if (wasActive) {
                const validNext = this._findFirstValidCandidate(setter);
                if (validNext) {
                    this._activateCandidate(setter, validNext);
                } else {
                    setter.properties.activeSourceId = null;
                    this.unassignSource(varName);
                }
            }

            app.graph.setDirtyCanvas(true, false);
            console.log(`[VariableManager] Removed candidate ${candidateId} from pool for "${varName}"`);
            return true;
        });
    }

    /**
     * Switch the active source for a variable to a different pool candidate.
     * Performs a real relink (unassign old → assign new) so wires update on canvas.
     */
    switchCandidate(varName, candidateId) {
        if (!app.graph) return false;

        return this._withUndo(() => {
            const setter = this._findSetter(varName);
            if (!setter || !setter.properties.sourceCandidates) return false;

            const candidate = setter.properties.sourceCandidates.find(c => c.id === candidateId);
            if (!candidate) {
                console.warn(`[VariableManager] Candidate ${candidateId} not found in pool for "${varName}"`);
                return false;
            }

            // Validate candidate is usable before attempting activation
            const sourceNode = app.graph.getNodeById(candidate.nodeId);
            if (!sourceNode) {
                console.warn(`[VariableManager] Cannot switch: source node ${candidate.nodeId} not found`);
                return false;
            }

            return this._activateCandidate(setter, candidate);
        });
    }

    /**
     * Get the source pool for a variable (read-only — does NOT mutate candidates).
     * Returns the candidates array with a `status` field added to each:
     * - "ok": node exists and type matches
     * - "stale": node no longer exists (and no fuzzy match found)
     * - "rebindable": node missing but a fuzzy match exists (call rebindCandidate to accept)
     * - "type_mismatch": node exists but output type changed
     */
    getPool(varName) {
        const setter = this._findSetter(varName);
        if (!setter || !setter.properties.sourceCandidates) return [];

        const varType = this._resolveType(setter);
        const activeId = setter.properties.activeSourceId;

        // Check if activeSourceId references a valid candidate (read-only — no mutation)
        const activeExists = activeId
            ? setter.properties.sourceCandidates.some(c => c.id === activeId)
            : false;

        return setter.properties.sourceCandidates.map(c => {
            const node = app.graph.getNodeById(c.nodeId);
            let status = "ok";
            let reboundNodeId = null;

            if (!node) {
                // Check for fuzzy match but do NOT mutate — return as suggestion
                const rebound = this._fuzzyRebind(c);
                if (rebound) {
                    status = "rebindable";
                    reboundNodeId = rebound.id;
                } else {
                    status = "stale";
                }
            } else {
                const outputSlot = node.outputs?.[c.slotIndex];
                if (!outputSlot) {
                    status = "stale";
                } else if (this._isTypeIncompatible(varType, outputSlot.type)) {
                    status = "type_mismatch";
                }
            }

            return {
                ...c,
                status,
                // Only mark as active if the activeSourceId still points to a real candidate
                isActive: activeExists && c.id === activeId,
                reboundNodeId,
            };
        });
    }

    /**
     * Accept a fuzzy rebind suggestion for a stale candidate.
     * Call this after getPool() returns a candidate with status "rebindable".
     */
    rebindCandidate(varName, candidateId, newNodeId) {
        return this._withUndo(() => {
            const setter = this._findSetter(varName);
            if (!setter?.properties?.sourceCandidates) return false;

            const candidate = setter.properties.sourceCandidates.find(c => c.id === candidateId);
            if (!candidate) return false;

            const node = app.graph.getNodeById(newNodeId);
            if (!node?.outputs?.[candidate.slotIndex]) return false;

            // Validate type compatibility
            const varType = this._resolveType(setter);
            const outputSlot = node.outputs[candidate.slotIndex];
            if (this._isTypeIncompatible(varType, outputSlot.type)) {
                console.warn(`[VariableManager] Rebind rejected: type mismatch (${varType} vs ${outputSlot.type})`);
                return false;
            }

            // Update all candidate fields to match the new node
            candidate.nodeId = newNodeId;
            candidate.nodeTitle = node.title || node.type;
            candidate.nodeType = node.type;
            candidate.outputType = outputSlot.type;

            // Re-activate if this was the active candidate
            if (setter.properties.activeSourceId === candidateId) {
                this._activateCandidate(setter, candidate);
            }

            app.graph.setDirtyCanvas(true, false);
            console.log(`[VariableManager] Rebound candidate "${candidate.label}" → node ${newNodeId}`);
            return true;
        });
    }

    /**
     * Rename a pool candidate's label.
     */
    renameCandidateLabel(varName, candidateId, newLabel) {
        return this._withUndo(() => {
            const setter = this._findSetter(varName);
            if (!setter || !setter.properties.sourceCandidates) return false;

            const candidate = setter.properties.sourceCandidates.find(c => c.id === candidateId);
            if (!candidate) return false;

            candidate.label = newLabel;
            app.graph.setDirtyCanvas(true, false);
            return true;
        });
    }

    /**
     * Remove all stale candidates (node deleted, slot missing) from a variable's pool.
     * Returns the number of entries purged.
     */
    purgeStale(varName) {
        return this._withUndo(() => {
            const setter = this._findSetter(varName);
            if (!setter || !setter.properties.sourceCandidates) return 0;

            const before = setter.properties.sourceCandidates.length;
            setter.properties.sourceCandidates = setter.properties.sourceCandidates.filter(c => {
                const node = app.graph.getNodeById(c.nodeId);
                if (!node) return false;
                if (!node.outputs?.[c.slotIndex]) return false;
                return true;
            });
            const purged = before - setter.properties.sourceCandidates.length;

            // If the active candidate was purged, fall back
            if (purged > 0 && setter.properties.activeSourceId) {
                const stillExists = setter.properties.sourceCandidates.some(
                    c => c.id === setter.properties.activeSourceId
                );
                if (!stillExists) {
                    const validNext = this._findFirstValidCandidate(setter);
                    if (validNext) {
                        this._activateCandidate(setter, validNext);
                    } else {
                        setter.properties.activeSourceId = null;
                        this.unassignSource(varName);
                    }
                }
            }

            if (purged > 0) {
                app.graph.setDirtyCanvas(true, false);
                console.log(`[VariableManager] Purged ${purged} stale candidate(s) from "${varName}"`);
            }
            return purged;
        });
    }

    /**
     * Purge stale candidates across ALL variables.
     * Returns total number of entries purged.
     */
    purgeAllStale() {
        return this._withUndo(() => {
            let total = 0;
            for (const name of this.getVariableNames()) {
                total += this.purgeStale(name);
            }
            return total;
        });
    }

    /**
     * Heal pool candidates across all variables on workflow load.
     * Fixes issues from older versions where _fuzzyRebind silently mutated
     * candidate nodeId/nodeTitle without updating other fields:
     * - Removes candidates pointing to nonexistent nodes
     * - Fixes stale nodeTitle/nodeType/outputType on valid candidates
     * - Clears activeSourceId if it references a removed candidate
     * - Cleans up orphaned sourceNodeId references on setters
     * Called once during migrateExistingSetters().
     * NOTE: Intentionally NOT wrapped in _withUndo() — this is a migration/repair
     * operation that runs on workflow load, not a user-interactive action.
     */
    healPool() {
        if (!app.graph) return;
        let totalHealed = 0;
        let totalPurged = 0;

        for (const name of this.getVariableNames()) {
            const setter = this._findSetter(name);
            if (!setter?.properties?.sourceCandidates) continue;

            const before = setter.properties.sourceCandidates.length;

            // Filter out truly dead candidates and fix stale metadata on living ones
            setter.properties.sourceCandidates = setter.properties.sourceCandidates.filter(c => {
                const node = app.graph.getNodeById(c.nodeId);
                if (!node) return false; // Purge — node is gone
                const outputSlot = node.outputs?.[c.slotIndex];
                if (!outputSlot) return false; // Purge — slot is gone

                // Heal stale metadata fields
                const currentTitle = node.title || node.type;
                let healed = false;
                if (c.nodeTitle !== currentTitle) {
                    c.nodeTitle = currentTitle;
                    healed = true;
                }
                if (c.nodeType !== node.type) {
                    c.nodeType = node.type;
                    healed = true;
                }
                if (c.outputType !== outputSlot.type) {
                    c.outputType = outputSlot.type;
                    healed = true;
                }
                if (healed) totalHealed++;
                return true;
            });

            totalPurged += before - setter.properties.sourceCandidates.length;

            // Validate activeSourceId
            if (setter.properties.activeSourceId) {
                const exists = setter.properties.sourceCandidates.some(
                    c => c.id === setter.properties.activeSourceId
                );
                if (!exists) {
                    setter.properties.activeSourceId = null;
                }
            }

            // Clean up orphaned sourceNodeId reference
            if (setter.properties.sourceNodeId) {
                const sourceNode = app.graph.getNodeById(setter.properties.sourceNodeId);
                if (!sourceNode) {
                    setter.properties.sourceNodeId = null;
                    setter.properties.sourceSlotIndex = null;
                }
            }
        }

        if (totalPurged > 0 || totalHealed > 0) {
            console.log(`[VariableManager] healPool: purged ${totalPurged} dead candidates, healed ${totalHealed} stale metadata entries`);
            app.graph.setDirtyCanvas(true, false);
        }
    }

    /**
     * Get all nodes with outputs compatible with a variable's type.
     * Used by the "Add to Pool" picker dropdown.
     */
    getCompatibleSources(varName) {
        if (!app.graph) return [];

        const setter = this._findSetter(varName);
        if (!setter) return [];

        const varType = this._resolveType(setter);
        const explicitType = setter.properties?.explicitType;
        const filterType = (explicitType && explicitType !== "*") ? explicitType : varType;
        const nodes = app.graph._nodes || [];
        const results = [];

        for (const node of nodes) {
            // Skip hidden/managed setter nodes
            if (node.properties?._nv_managed) continue;
            if (!node.outputs) continue;

            for (let i = 0; i < node.outputs.length; i++) {
                const output = node.outputs[i];
                if (!output || !output.type) continue;

                // Match if wildcard or type matches
                if (filterType === "*" || filterType === "unconnected" || output.type === "*" || output.type === filterType) {
                    const nodeTitle = node.title || node.type;
                    const outputName = output.name || `output_${i}`;
                    results.push({
                        nodeId: node.id,
                        nodeTitle,
                        nodeType: node.type,
                        slotIndex: i,
                        outputType: output.type,
                        outputName,
                        label: `${nodeTitle}:${outputName}`,
                    });
                }
            }
        }

        return results;
    }

    /** @private Activate a pool candidate — performs real relink */
    _activateCandidate(setter, candidate) {
        if (!setter?.inputs?.[0]) return false;

        const sourceNode = app.graph.getNodeById(candidate.nodeId);
        if (!sourceNode) {
            console.warn(`[VariableManager] Cannot activate: source node ${candidate.nodeId} not found`);
            return false;
        }

        // Validate output slot exists
        const outputSlot = sourceNode.outputs?.[candidate.slotIndex];
        if (!outputSlot) {
            console.warn(`[VariableManager] Cannot activate: slot ${candidate.slotIndex} missing on node ${candidate.nodeId}`);
            return false;
        }

        // Validate type compatibility
        const varType = this._resolveType(setter);
        if (this._isTypeIncompatible(varType, outputSlot.type)) {
            console.warn(`[VariableManager] Cannot activate: type mismatch (${varType} vs ${outputSlot.type})`);
            return false;
        }

        const varName = setter.widgets?.[0]?.value;

        // Remove existing input link
        if (setter.inputs[0].link != null) {
            app.graph.removeLink(setter.inputs[0].link);
        }

        // Create new link
        sourceNode.connect(candidate.slotIndex, setter, 0);

        // Update tracking
        setter.properties.activeSourceId = candidate.id;
        setter.properties.sourceNodeId = sourceNode.id;
        setter.properties.sourceSlotIndex = candidate.slotIndex;

        // Propagate type to getters
        if (setter.updateGetters) {
            setter.updateGetters();
        }

        app.graph.setDirtyCanvas(true, false);
        console.log(`[VariableManager] Activated "${candidate.label}" for "${varName}"`);
        return true;
    }

    /** @private Find first valid (non-stale, type-compatible) candidate in pool */
    _findFirstValidCandidate(setter) {
        if (!setter.properties?.sourceCandidates) return null;
        const varType = this._resolveType(setter);

        for (const c of setter.properties.sourceCandidates) {
            const node = app.graph.getNodeById(c.nodeId);
            if (!node) continue;
            const outputSlot = node.outputs?.[c.slotIndex];
            if (!outputSlot) continue;
            if (this._isTypeIncompatible(varType, outputSlot.type)) continue;
            return c;
        }
        return null;
    }

    /**
     * @private Attempt to find a fuzzy match for a stale candidate by nodeType + title + output type.
     * Returns the matching node or null. Does NOT mutate the candidate.
     */
    _fuzzyRebind(candidate) {
        if (!app.graph) return null;
        const nodes = app.graph._nodes || [];
        let match = null;
        let matchCount = 0;

        for (const node of nodes) {
            if (node.type !== candidate.nodeType) continue;
            const title = node.title || node.type;
            if (title !== candidate.nodeTitle) continue;
            const outputSlot = node.outputs?.[candidate.slotIndex];
            if (!outputSlot) continue;
            // Validate output type matches what the candidate originally had
            if (candidate.outputType && outputSlot.type !== candidate.outputType) continue;

            match = node;
            matchCount++;
        }

        // Only return if exactly one match — ambiguous matches are unsafe
        if (matchCount === 1) return match;
        if (matchCount > 1) {
            console.warn(`[VariableManager] Fuzzy rebind ambiguous: ${matchCount} nodes match "${candidate.nodeTitle}" (${candidate.nodeType})`);
        }
        return null;
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
        // Build a set of live node IDs once for O(1) staleness checks
        const liveNodeIds = new Set(nodes.map(n => n.id));
        let hash = "";
        for (const n of nodes) {
            if (n.type === "SetVariableNode" || n.type === "GetVariableNode") {
                let part = `${n.id}=${n.widgets?.[0]?.value}|${n.inputs?.[0]?.type}|${n.inputs?.[0]?.link}|${n.properties?.explicitType}|${n.properties?.activeSourceId || ""}`;
                // Include pool candidate count + alive count so deletions trigger refresh
                const candidates = n.properties?.sourceCandidates;
                if (candidates && candidates.length > 0) {
                    const aliveCount = candidates.filter(c => liveNodeIds.has(c.nodeId)).length;
                    part += `|pool:${candidates.length}/${aliveCount}`;
                }
                hash += part + ",";
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

        // graph._links is a Map in ComfyUI frontend >= 1.10 — use .get()
        const linkId = setter.inputs[0].link;
        const link = app.graph._links?.get(linkId)
            ?? app.graph.links?.[linkId]
            ?? null;
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

        // Heal pool candidates — fix orphaned refs from older versions
        this.healPool();
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

    /** @private Check if two types are incompatible (neither is wildcard/unconnected) */
    _isTypeIncompatible(varType, slotType) {
        if (varType === "*" || varType === "unconnected") return false;
        if (slotType === "*") return false;
        return slotType !== varType;
    }

    /** @private Wrap a graph-mutating operation in LiteGraph undo tracking.
     *  Reentrant-safe: nested calls skip the bracket so only one undo entry is created. */
    _withUndo(fn) {
        if (this._inUndo) return fn(); // Already inside a transaction
        this._inUndo = true;
        if (app.graph?.beforeChange) app.graph.beforeChange();
        try {
            return fn();
        } finally {
            if (app.graph?.afterChange) app.graph.afterChange();
            this._inUndo = false;
        }
    }

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
