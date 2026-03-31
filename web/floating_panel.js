/**
 * Floating Panel - Viewport-fixed group/node muter/bypasser with Macro Groups
 *
 * A panel that stays fixed to the viewport (doesn't move when panning)
 * and provides quick access to mute/bypass groups and node patterns.
 *
 * Features:
 * - Viewport-fixed positioning (stays in place during pan/zoom)
 * - Draggable with position persistence
 * - Collapsible to minimize screen space
 * - Auto-discovers groups like rgthree's fast groups muter
 * - Supports custom node patterns for bypass control
 * - Macro Groups: bundle multiple groups into one toggle
 */

import { app } from "../../scripts/app.js";

const MODE_ALWAYS = 0;
const MODE_BYPASS = 4;
const MODE_NEVER = 2;

const STORAGE_KEY = "NV_FloatingPanel_State";
const MACRO_STORAGE_KEY = "NV_FloatingPanel_Macros";
const DEFAULT_POSITION = { x: 20, y: 100 };
const STATE_VERSION = 1;

// ─── MacroManager ─────────────────────────────────────────────────────────────
// Handles CRUD, state computation, and serialization for macro groups.
// Never touches DOM or ComfyUI globals directly — the panel mediates.

class MacroManager {
    constructor() {
        this.macros = [];
        this.macroOrder = [];
        this._load();
    }

    _load() {
        try {
            const saved = localStorage.getItem(MACRO_STORAGE_KEY);
            if (saved) {
                const data = JSON.parse(saved);
                this.macros = Array.isArray(data.macros) ? data.macros : [];
                this.macroOrder = Array.isArray(data.macroOrder) ? data.macroOrder : [];
                // Migration: ensure all macros have required fields
                for (const m of this.macros) {
                    if (!m.id) m.id = "m_" + Date.now() + "_" + Math.random().toString(36).slice(2, 6);
                    if (!Array.isArray(m.groupTitles)) m.groupTitles = [];
                    if (typeof m.collapsed !== "boolean") m.collapsed = false;
                    if (typeof m.name !== "string") m.name = "Untitled Macro";
                }
                // Prune macroOrder to only valid IDs
                const idSet = new Set(this.macros.map(m => m.id));
                this.macroOrder = this.macroOrder.filter(id => idSet.has(id));
                // Add any macros not in order
                for (const m of this.macros) {
                    if (!this.macroOrder.includes(m.id)) this.macroOrder.push(m.id);
                }
            }
        } catch (e) {
            console.warn("[MacroManager] Failed to load macros, resetting:", e);
            this.macros = [];
            this.macroOrder = [];
        }
    }

    _save() {
        try {
            localStorage.setItem(MACRO_STORAGE_KEY, JSON.stringify({
                version: STATE_VERSION,
                macros: this.macros,
                macroOrder: this.macroOrder,
            }));
        } catch (e) {
            console.warn("[MacroManager] Failed to save macros:", e);
        }
    }

    getMacros() {
        // Return macros in display order
        const byId = new Map(this.macros.map(m => [m.id, m]));
        return this.macroOrder.map(id => byId.get(id)).filter(Boolean);
    }

    createMacro(name, groupTitles) {
        const macro = {
            id: "m_" + Date.now() + "_" + Math.random().toString(36).slice(2, 6),
            name,
            groupTitles: [...groupTitles],
            collapsed: false,
        };
        this.macros.push(macro);
        this.macroOrder.push(macro.id);
        this._save();
        return macro;
    }

    updateMacro(id, changes) {
        const macro = this.macros.find(m => m.id === id);
        if (!macro) return null;
        if (changes.name !== undefined) macro.name = changes.name;
        if (changes.groupTitles !== undefined) macro.groupTitles = [...changes.groupTitles];
        if (changes.collapsed !== undefined) macro.collapsed = changes.collapsed;
        this._save();
        return macro;
    }

    deleteMacro(id) {
        this.macros = this.macros.filter(m => m.id !== id);
        this.macroOrder = this.macroOrder.filter(i => i !== id);
        this._save();
    }

    reorderMacros(newOrder) {
        this.macroOrder = newOrder;
        this._save();
    }

    /**
     * Compute tri-state for a macro given live groups.
     * @param {object} macro
     * @param {Map<string, object[]>} groupMap - title -> [group objects]
     * @returns {{ state: 'all'|'partial'|'none', enabled: number, total: number, stale: string[] }}
     */
    computeTriState(macro, groupMap) {
        let enabled = 0;
        let total = 0;
        const stale = [];

        for (const title of macro.groupTitles) {
            const groups = groupMap.get(title);
            if (!groups || groups.length === 0) {
                stale.push(title);
                continue;
            }
            for (const group of groups) {
                const nodes = group._nodes || [];
                if (nodes.length === 0) continue;
                total++;
                const allActive = nodes.every(n => n.mode === MODE_ALWAYS);
                if (allActive) enabled++;
            }
        }

        let state;
        if (total === 0) state = "none";
        else if (enabled === total) state = "all";
        else if (enabled === 0) state = "none";
        else state = "partial";

        return { state, enabled, total, stale };
    }

    /** Get set of all group titles claimed by any macro */
    getClaimedTitles() {
        const claimed = new Set();
        for (const m of this.macros) {
            for (const t of m.groupTitles) claimed.add(t);
        }
        return claimed;
    }
}

// ─── FloatingPanel ────────────────────────────────────────────────────────────

class FloatingPanel {
    constructor() {
        this.container = null;
        this.header = null;
        this.content = null;
        this.groupsList = null;
        this.patternsList = null;
        this.isCollapsed = false;
        this.isDragging = false;
        this.dragOffset = { x: 0, y: 0 };
        this.position = { ...DEFAULT_POSITION };
        this.customPatterns = [];
        this.groupOrder = []; // Persisted array of group titles for manual ordering
        this.isDraggingGroup = false; // True during group row drag-and-drop
        this.refreshInterval = null;
        this.isVisible = true;
        this._dialogOpen = false; // P0: pause polling during dialogs
        this._activeDialogs = []; // P1: track open dialogs for cleanup
        this._lastStateHash = ""; // Performance: dirty-check to skip DOM rebuilds

        // Macro manager (error-isolated — P1: corrupt state won't kill the panel)
        try {
            this.macroManager = new MacroManager();
        } catch (e) {
            console.error("[FloatingPanel] MacroManager init failed, macros disabled:", e);
            this.macroManager = null;
        }

        this.loadState();
        this.createPanel();
        this.startRefreshLoop();
    }

    loadState() {
        try {
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved) {
                const state = JSON.parse(saved);
                this.position = state.position || { ...DEFAULT_POSITION };
                this.isCollapsed = state.isCollapsed || false;
                this.customPatterns = state.customPatterns || [];
                this.groupOrder = state.groupOrder || [];
                this.isVisible = state.isVisible !== false;
            }
        } catch (e) {
            console.warn("[FloatingPanel] Failed to load state:", e);
        }
    }

    saveState() {
        try {
            const state = {
                version: STATE_VERSION,
                position: this.position,
                isCollapsed: this.isCollapsed,
                customPatterns: this.customPatterns,
                groupOrder: this.groupOrder,
                isVisible: this.isVisible
            };
            localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        } catch (e) {
            console.warn("[FloatingPanel] Failed to save state:", e);
        }
    }

    createPanel() {
        // Main container - viewport fixed
        this.container = document.createElement("div");
        this.container.id = "nv-floating-panel";
        this.container.style.cssText = `
            position: fixed;
            left: ${this.position.x}px;
            top: ${this.position.y}px;
            width: 280px;
            background: #1a1a1a;
            border: 1px solid #444;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            z-index: 100001;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 12px;
            color: #e0e0e0;
            user-select: none;
            pointer-events: auto;
            display: ${this.isVisible ? 'block' : 'none'};
        `;

        // Header (draggable)
        this.header = document.createElement("div");
        this.header.style.cssText = `
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 12px;
            background: #2a2a2a;
            border-radius: 7px 7px 0 0;
            cursor: move;
            border-bottom: 1px solid #444;
        `;

        const title = document.createElement("span");
        title.textContent = "Quick Toggle";
        title.style.fontWeight = "600";

        const headerButtons = document.createElement("div");
        headerButtons.style.cssText = "display: flex; gap: 4px;";

        // Refresh button
        const refreshBtn = this.createIconButton("↻", "Refresh groups", () => this.refreshGroups());

        // Settings button
        const settingsBtn = this.createIconButton("⚙", "Add pattern", () => this.showAddPatternDialog());

        // Collapse button
        this.collapseBtn = this.createIconButton(this.isCollapsed ? "▼" : "▲", "Toggle collapse", () => this.toggleCollapse());

        // Close button
        const closeBtn = this.createIconButton("×", "Hide panel", () => this.hide());
        closeBtn.style.fontSize = "16px";

        headerButtons.appendChild(refreshBtn);
        headerButtons.appendChild(settingsBtn);
        headerButtons.appendChild(this.collapseBtn);
        headerButtons.appendChild(closeBtn);

        this.header.appendChild(title);
        this.header.appendChild(headerButtons);

        // Content area
        this.content = document.createElement("div");
        this.content.style.cssText = `
            max-height: 400px;
            overflow-y: auto;
            display: ${this.isCollapsed ? 'none' : 'block'};
        `;

        // Groups section (with reset order button) — will contain macros + ungrouped
        const groupsSection = this.createSection("Groups", "groupsList");
        this.groupsList = groupsSection.list;

        const resetOrderBtn = this.createIconButton("⟳", "Reset to position order", () => {
            this.groupOrder = [];
            this.saveState();
            this.refreshGroups();
        });
        resetOrderBtn.style.fontSize = "11px";
        resetOrderBtn.style.padding = "0 4px";
        groupsSection.header.appendChild(resetOrderBtn);

        // Custom patterns section
        const patternsSection = this.createSection("Custom Patterns", "patternsList");
        this.patternsList = patternsSection.list;

        // Bulk actions
        const bulkActions = document.createElement("div");
        bulkActions.style.cssText = `
            display: flex;
            gap: 4px;
            padding: 8px;
            border-top: 1px solid #333;
            background: #222;
            border-radius: 0 0 7px 7px;
        `;

        const muteAllBtn = this.createActionButton("Mute All", "#a55", () => this.setAllGroupsMode(MODE_NEVER));
        const enableAllBtn = this.createActionButton("Enable All", "#5a5", () => this.setAllGroupsMode(MODE_ALWAYS));
        const bypassAllBtn = this.createActionButton("Bypass All", "#55a", () => this.setAllGroupsMode(MODE_BYPASS));

        bulkActions.appendChild(muteAllBtn);
        bulkActions.appendChild(enableAllBtn);
        bulkActions.appendChild(bypassAllBtn);

        this.content.appendChild(groupsSection.container);
        this.content.appendChild(patternsSection.container);

        this.container.appendChild(this.header);
        this.container.appendChild(this.content);
        this.container.appendChild(bulkActions);

        document.body.appendChild(this.container);

        // Set up drag handlers
        this.setupDragHandlers();

        // Initial refresh
        this.refreshGroups();
        this.refreshPatterns();
    }

    createIconButton(icon, tooltip, onClick) {
        const btn = document.createElement("button");
        btn.textContent = icon;
        btn.title = tooltip;
        btn.style.cssText = `
            background: transparent;
            border: none;
            color: #888;
            cursor: pointer;
            padding: 2px 6px;
            font-size: 14px;
            border-radius: 3px;
            transition: all 0.15s;
        `;
        btn.addEventListener("mouseenter", () => btn.style.background = "#444");
        btn.addEventListener("mouseleave", () => btn.style.background = "transparent");
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            onClick();
        });
        return btn;
    }

    createActionButton(text, color, onClick) {
        const btn = document.createElement("button");
        btn.textContent = text;
        btn.style.cssText = `
            flex: 1;
            padding: 6px 8px;
            background: ${color};
            border: none;
            border-radius: 4px;
            color: white;
            cursor: pointer;
            font-size: 11px;
            font-weight: 500;
            transition: filter 0.15s;
        `;
        btn.addEventListener("mouseenter", () => btn.style.filter = "brightness(1.2)");
        btn.addEventListener("mouseleave", () => btn.style.filter = "none");
        btn.addEventListener("click", onClick);
        return btn;
    }

    createSection(title, listId) {
        const container = document.createElement("div");
        container.style.cssText = "margin-bottom: 4px;";

        const header = document.createElement("div");
        header.style.cssText = `
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 6px 10px;
            background: #252525;
            font-size: 11px;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        `;
        const headerLabel = document.createElement("span");
        headerLabel.textContent = title;
        header.appendChild(headerLabel);

        const list = document.createElement("div");
        list.id = listId;
        list.style.cssText = "padding: 4px;";

        container.appendChild(header);
        container.appendChild(list);

        return { container, header, list };
    }

    createToggleRow(name, isActive, onToggle, onNavigate = null, onRemove = null) {
        const row = document.createElement("div");
        row.style.cssText = `
            display: flex;
            align-items: center;
            padding: 6px 8px;
            margin: 2px 0;
            background: #2a2a2a;
            border-radius: 4px;
            transition: background 0.15s;
        `;
        row.addEventListener("mouseenter", () => row.style.background = "#333");
        row.addEventListener("mouseleave", () => row.style.background = "#2a2a2a");

        // Toggle indicator
        const toggle = document.createElement("div");
        toggle.style.cssText = `
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: ${isActive ? '#5a5' : '#555'};
            margin-right: 8px;
            cursor: pointer;
            transition: background 0.15s;
            flex-shrink: 0;
        `;
        toggle.addEventListener("click", () => {
            onToggle(!isActive);
        });

        // Name (always textContent — P0: XSS safe)
        const label = document.createElement("span");
        label.textContent = name;
        label.style.cssText = `
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            cursor: pointer;
        `;
        label.addEventListener("click", () => onToggle(!isActive));

        row.appendChild(toggle);
        row.appendChild(label);

        // Navigate button (for groups)
        if (onNavigate) {
            const navBtn = document.createElement("button");
            navBtn.textContent = "→";
            navBtn.title = "Go to group";
            navBtn.style.cssText = `
                background: transparent;
                border: none;
                color: #666;
                cursor: pointer;
                padding: 2px 6px;
                font-size: 12px;
                margin-left: 4px;
            `;
            navBtn.addEventListener("mouseenter", () => navBtn.style.color = "#aaa");
            navBtn.addEventListener("mouseleave", () => navBtn.style.color = "#666");
            navBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                onNavigate();
            });
            row.appendChild(navBtn);
        }

        // Remove button (for custom patterns)
        if (onRemove) {
            const removeBtn = document.createElement("button");
            removeBtn.textContent = "×";
            removeBtn.title = "Remove pattern";
            removeBtn.style.cssText = `
                background: transparent;
                border: none;
                color: #666;
                cursor: pointer;
                padding: 2px 6px;
                font-size: 14px;
                margin-left: 4px;
            `;
            removeBtn.addEventListener("mouseenter", () => removeBtn.style.color = "#a55");
            removeBtn.addEventListener("mouseleave", () => removeBtn.style.color = "#666");
            removeBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                onRemove();
            });
            row.appendChild(removeBtn);
        }

        return row;
    }

    setupDragHandlers() {
        this.header.addEventListener("mousedown", (e) => {
            if (e.target.tagName === "BUTTON") return;
            this.isDragging = true;
            this.dragOffset = {
                x: e.clientX - this.position.x,
                y: e.clientY - this.position.y
            };
            e.preventDefault();
        });

        document.addEventListener("mousemove", (e) => {
            if (!this.isDragging) return;

            this.position.x = Math.max(0, Math.min(window.innerWidth - 300, e.clientX - this.dragOffset.x));
            this.position.y = Math.max(0, Math.min(window.innerHeight - 100, e.clientY - this.dragOffset.y));

            this.container.style.left = `${this.position.x}px`;
            this.container.style.top = `${this.position.y}px`;
        });

        document.addEventListener("mouseup", () => {
            if (this.isDragging) {
                this.isDragging = false;
                this.saveState();
            }
        });
    }

    toggleCollapse() {
        this.isCollapsed = !this.isCollapsed;
        this.content.style.display = this.isCollapsed ? 'none' : 'block';
        this.collapseBtn.textContent = this.isCollapsed ? "▼" : "▲";
        this.saveState();
    }

    show() {
        this.isVisible = true;
        this.container.style.display = 'block';

        // Reset position if off-screen
        const maxX = window.innerWidth - 50;
        const maxY = window.innerHeight - 50;
        if (this.position.x < 0 || this.position.x > maxX || this.position.y < 0 || this.position.y > maxY) {
            this.position = { ...DEFAULT_POSITION };
            this.container.style.left = `${this.position.x}px`;
            this.container.style.top = `${this.position.y}px`;
        }

        this.saveState();
        this.refreshGroups();
    }

    hide() {
        this.isVisible = false;
        this.container.style.display = 'none';
        // P1: clean up any open dialogs
        this._closeAllDialogs();
        this.saveState();
    }

    toggle() {
        if (this.isVisible) {
            this.hide();
        } else {
            this.show();
        }
    }

    // ─── Dirty-check: skip DOM rebuild if nothing changed ──────────────

    _computeStateHash(groups) {
        // Build a hash of group titles + modes + macro definitions
        // O(N) string ops — sub-millisecond even for 100+ groups
        let hash = "";
        for (const g of groups) {
            const title = g.title || "";
            const modes = (g._nodes || []).map(n => n.mode).join("");
            hash += title + ":" + modes + "|";
        }
        if (this.macroManager) {
            for (const m of this.macroManager.macros) {
                hash += "M" + m.id + ":" + m.name + ":" + m.groupTitles.join(",") + ":" + (m.collapsed ? "1" : "0") + "|";
            }
            hash += "O:" + this.macroManager.macroOrder.join(",");
        }
        return hash;
    }

    // ─── Build group lookup map ────────────────────────────────────────

    _buildGroupMap(groups) {
        const groupMap = new Map();
        for (const g of groups) {
            const title = g.title || "Untitled Group";
            if (!groupMap.has(title)) groupMap.set(title, []);
            groupMap.get(title).push(g);
        }
        return groupMap;
    }

    // ─── Sort groups helper ────────────────────────────────────────────

    _sortGroups(groups) {
        if (this.groupOrder.length > 0) {
            const orderMap = new Map(this.groupOrder.map((title, i) => [title, i]));
            return [...groups].sort((a, b) => {
                const aTitle = a.title || "Untitled Group";
                const bTitle = b.title || "Untitled Group";
                const aIdx = orderMap.has(aTitle) ? orderMap.get(aTitle) : Infinity;
                const bIdx = orderMap.has(bTitle) ? orderMap.get(bTitle) : Infinity;
                if (aIdx !== bIdx) return aIdx - bIdx;
                const aY = Math.floor(a._pos[1] / 30);
                const bY = Math.floor(b._pos[1] / 30);
                if (aY === bY) return Math.floor(a._pos[0] / 30) - Math.floor(b._pos[0] / 30);
                return aY - bY;
            });
        }
        return [...groups].sort((a, b) => {
            const aY = Math.floor(a._pos[1] / 30);
            const bY = Math.floor(b._pos[1] / 30);
            if (aY === bY) return Math.floor(a._pos[0] / 30) - Math.floor(b._pos[0] / 30);
            return aY - bY;
        });
    }

    // ─── Main refresh (macros + ungrouped + patterns) ──────────────────

    refreshGroups() {
        if (!app.graph || this.isDraggingGroup) return;

        // Detect graph object swap (tab switch, workflow load, etc.)
        if (this._lastGraph !== app.graph) {
            this._lastGraph = app.graph;
            this._lastStateHash = "";
        }

        const groups = app.graph._groups || [];

        // Recompute nodes for all groups
        for (const group of groups) {
            if (group.recomputeInsideNodes) group.recomputeInsideNodes();
        }

        // Dirty-check: skip DOM rebuild if nothing changed
        const stateHash = this._computeStateHash(groups);
        if (stateHash === this._lastStateHash) return;
        this._lastStateHash = stateHash;

        this.groupsList.innerHTML = "";

        if (groups.length === 0) {
            const empty = document.createElement("div");
            empty.textContent = "No groups in workflow";
            empty.style.cssText = "padding: 8px; color: #666; font-style: italic; text-align: center;";
            this.groupsList.appendChild(empty);
            return;
        }

        const groupMap = this._buildGroupMap(groups);
        const sortedGroups = this._sortGroups(groups);

        // ── Macros section ──
        if (this.macroManager && this.macroManager.macros.length > 0) {
            const macroHeader = document.createElement("div");
            macroHeader.style.cssText = `
                display: flex; align-items: center; justify-content: space-between;
                padding: 4px 8px; font-size: 10px; color: #777; text-transform: uppercase;
                letter-spacing: 0.5px;
            `;
            const macroLabel = document.createElement("span");
            macroLabel.textContent = "Macros";
            macroHeader.appendChild(macroLabel);

            const addMacroBtn = this.createIconButton("+", "Create macro group", () => this.showMacroDialog());
            addMacroBtn.style.fontSize = "13px";
            addMacroBtn.style.padding = "0 4px";
            macroHeader.appendChild(addMacroBtn);
            this.groupsList.appendChild(macroHeader);

            const macros = this.macroManager.getMacros();
            for (const macro of macros) {
                this._renderMacro(macro, groupMap);
            }

            // ── Ungrouped separator ──
            const claimedTitles = this.macroManager.getClaimedTitles();
            const ungrouped = sortedGroups.filter(g => !claimedTitles.has(g.title || "Untitled Group"));

            if (ungrouped.length > 0) {
                const ungroupedHeader = document.createElement("div");
                ungroupedHeader.style.cssText = `
                    padding: 4px 8px; font-size: 10px; color: #777; text-transform: uppercase;
                    letter-spacing: 0.5px; margin-top: 4px;
                `;
                ungroupedHeader.textContent = "Groups";
                this.groupsList.appendChild(ungroupedHeader);

                for (const group of ungrouped) {
                    this._renderGroupRow(group, true);
                }
            }
        } else {
            // No macros — render add button + all groups flat
            if (this.macroManager) {
                const addMacroBtn = this.createIconButton("+", "Create macro group", () => this.showMacroDialog());
                addMacroBtn.style.fontSize = "13px";
                addMacroBtn.style.padding = "0 4px";
                addMacroBtn.style.float = "right";
                this.groupsList.appendChild(addMacroBtn);
            }

            for (const group of sortedGroups) {
                this._renderGroupRow(group, true);
            }
        }
    }

    // ─── Render a single macro with its children ───────────────────────

    _renderMacro(macro, groupMap) {
        const triState = this.macroManager.computeTriState(macro, groupMap);

        // Macro header row
        const macroRow = document.createElement("div");
        const bgColor = triState.state === "all" ? "#1e2e1e" :
                        triState.state === "none" ? "#2e1e1e" : "#2e2e1e";
        macroRow.style.cssText = `
            display: flex; align-items: center; padding: 6px 8px; margin: 2px 0;
            background: ${bgColor}; border-radius: 4px; transition: background 0.15s;
            border-left: 3px solid ${triState.state === "all" ? "#5a5" : triState.state === "partial" ? "#aa5" : "#555"};
        `;
        macroRow.addEventListener("mouseenter", () => { macroRow.style.filter = "brightness(1.15)"; });
        macroRow.addEventListener("mouseleave", () => { macroRow.style.filter = "none"; });

        // Tri-state toggle indicator
        const toggle = document.createElement("div");
        const toggleColor = triState.state === "all" ? "#5a5" :
                           triState.state === "partial" ? "#aa5" : "#555";
        toggle.style.cssText = `
            width: 12px; height: 12px; border-radius: 50%; background: ${toggleColor};
            margin-right: 8px; cursor: pointer; flex-shrink: 0;
        `;
        if (triState.state === "partial") {
            // Half-filled effect for partial state
            toggle.style.background = `linear-gradient(90deg, #5a5 50%, #555 50%)`;
        }
        toggle.addEventListener("click", () => this.toggleMacro(macro, groupMap));
        macroRow.appendChild(toggle);

        // Macro name
        const label = document.createElement("span");
        label.textContent = macro.name;
        label.style.cssText = `
            flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
            cursor: pointer; font-weight: 600;
        `;
        label.addEventListener("click", () => this.toggleMacro(macro, groupMap));
        macroRow.appendChild(label);

        // Count badge
        const badge = document.createElement("span");
        badge.textContent = `${triState.enabled}/${triState.total}`;
        badge.style.cssText = `
            font-size: 10px; color: #888; margin-left: 4px; margin-right: 4px; flex-shrink: 0;
        `;
        macroRow.appendChild(badge);

        // Stale warning
        if (triState.stale.length > 0) {
            const warnBtn = document.createElement("span");
            warnBtn.textContent = "!";
            warnBtn.title = "Missing groups: " + triState.stale.join(", ");
            warnBtn.style.cssText = `
                color: #a55; font-weight: bold; font-size: 12px; margin-right: 4px; cursor: help;
            `;
            macroRow.appendChild(warnBtn);
        }

        // Collapse chevron
        const chevron = this.createIconButton(macro.collapsed ? "▶" : "▼", "Expand/collapse", () => {
            this.macroManager.updateMacro(macro.id, { collapsed: !macro.collapsed });
            this._lastStateHash = ""; // force refresh
            this.refreshGroups();
        });
        chevron.style.fontSize = "10px";
        chevron.style.padding = "0 3px";
        macroRow.appendChild(chevron);

        // Edit button (visible on hover via CSS-in-JS)
        const editBtn = this.createIconButton("✎", "Edit macro", () => this.showMacroDialog(macro));
        editBtn.style.fontSize = "11px";
        editBtn.style.padding = "0 3px";
        macroRow.appendChild(editBtn);

        this.groupsList.appendChild(macroRow);

        // ── Child group rows (if expanded) ──
        if (!macro.collapsed) {
            const childContainer = document.createElement("div");
            childContainer.style.cssText = "padding-left: 16px; border-left: 1px solid #444; margin-left: 14px;";

            for (const title of macro.groupTitles) {
                const matchingGroups = groupMap.get(title);
                if (!matchingGroups || matchingGroups.length === 0) {
                    // Stale reference row
                    const staleRow = document.createElement("div");
                    staleRow.style.cssText = `
                        display: flex; align-items: center; padding: 4px 8px; margin: 1px 0;
                        background: #2a2222; border-radius: 4px; color: #777;
                    `;
                    const staleLabel = document.createElement("span");
                    staleLabel.textContent = title;
                    staleLabel.style.cssText = "flex: 1; text-decoration: line-through; font-style: italic;";
                    staleRow.appendChild(staleLabel);
                    childContainer.appendChild(staleRow);
                    continue;
                }

                // Render each matching group (usually 1, but handles duplicates)
                for (const group of matchingGroups) {
                    this._renderGroupRow(group, false, childContainer);
                }
            }
            this.groupsList.appendChild(childContainer);
        }
    }

    // ─── Render a single group row ─────────────────────────────────────

    _renderGroupRow(group, enableDrag, appendTarget = null) {
        const hasActiveNodes = (group._nodes || []).some(n => n.mode === MODE_ALWAYS);
        const groupTitle = group.title || "Untitled Group";

        const row = this.createToggleRow(
            groupTitle,
            hasActiveNodes,
            (enable) => this.toggleGroup(group, enable),
            () => this.navigateToGroup(group)
        );

        // Add color indicator matching group color
        if (group.color) {
            row.style.borderLeft = `3px solid ${group.color}`;
        }

        if (enableDrag) {
            // Make row draggable for reordering (P1: only for top-level, not macro children)
            row.draggable = true;
            row.dataset.groupTitle = groupTitle;
            row.style.cursor = "grab";

            row.addEventListener("dragstart", (e) => {
                this.isDraggingGroup = true;
                this._draggedGroupTitle = groupTitle;
                row.style.opacity = "0.4";
                e.dataTransfer.effectAllowed = "move";
                e.dataTransfer.setData("text/plain", groupTitle);
            });

            row.addEventListener("dragend", () => {
                this.isDraggingGroup = false;
                row.style.opacity = "1";
                this.groupsList.querySelectorAll("[data-group-title]").forEach(el => {
                    el.style.borderTop = "";
                    el.style.borderBottom = "";
                });
            });

            row.addEventListener("dragover", (e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = "move";
                const rect = row.getBoundingClientRect();
                const midY = rect.top + rect.height / 2;
                this.groupsList.querySelectorAll("[data-group-title]").forEach(el => {
                    el.style.borderTop = "";
                    el.style.borderBottom = "";
                });
                if (e.clientY < midY) {
                    row.style.borderTop = "2px solid #5af";
                } else {
                    row.style.borderBottom = "2px solid #5af";
                }
            });

            row.addEventListener("dragleave", () => {
                row.style.borderTop = "";
                row.style.borderBottom = "";
            });

            row.addEventListener("drop", (e) => {
                e.preventDefault();
                row.style.borderTop = "";
                row.style.borderBottom = "";

                const draggedTitle = this._draggedGroupTitle;
                const targetTitle = groupTitle;
                if (!draggedTitle || draggedTitle === targetTitle) return;

                const currentOrder = Array.from(
                    this.groupsList.querySelectorAll("[data-group-title]")
                ).map(el => el.dataset.groupTitle);

                const fromIdx = currentOrder.indexOf(draggedTitle);
                const toIdx = currentOrder.indexOf(targetTitle);
                if (fromIdx === -1 || toIdx === -1) return;

                const rect2 = row.getBoundingClientRect();
                const insertAfter = e.clientY >= rect2.top + rect2.height / 2;

                currentOrder.splice(fromIdx, 1);
                let insertIdx = currentOrder.indexOf(targetTitle);
                if (insertAfter) insertIdx++;
                currentOrder.splice(insertIdx, 0, draggedTitle);

                this.groupOrder = currentOrder;
                this.saveState();
                this.isDraggingGroup = false;
                this.refreshGroups();
            });
        }

        (appendTarget || this.groupsList).appendChild(row);
    }

    // ─── Macro toggle ──────────────────────────────────────────────────

    toggleMacro(macro, groupMap) {
        const triState = this.macroManager.computeTriState(macro, groupMap);
        // any-on → turn all off; all-off → turn all on
        const targetMode = (triState.state === "none") ? MODE_ALWAYS : MODE_BYPASS;

        // P0: Wrap in undo transaction (Gemini review finding)
        if (app.graph.beforeChange) app.graph.beforeChange();

        for (const title of macro.groupTitles) {
            const groups = groupMap.get(title);
            if (!groups) continue;
            for (const group of groups) {
                for (const node of group._nodes || []) {
                    node.mode = targetMode;
                }
            }
        }

        if (app.graph.afterChange) app.graph.afterChange();
        app.graph.setDirtyCanvas(true, false);
        this._lastStateHash = ""; // force refresh
        this.refreshGroups();
    }

    // ─── Macro dialog (create / edit) ──────────────────────────────────

    showMacroDialog(existingMacro = null) {
        this._dialogOpen = true; // P0: pause polling

        const isEdit = !!existingMacro;
        const groups = app.graph?._groups || [];

        // Overlay
        const overlay = document.createElement("div");
        overlay.style.cssText = `
            position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 100002;
        `;

        // Dialog
        const dialog = document.createElement("div");
        dialog.style.cssText = `
            position: fixed; left: 50%; top: 50%; transform: translate(-50%, -50%);
            background: #1a1a1a; border: 1px solid #444; border-radius: 8px; padding: 16px;
            z-index: 100003; min-width: 320px; max-width: 400px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
        `;

        const closeDialog = () => {
            this._dialogOpen = false;
            if (dialog.parentElement) dialog.parentElement.removeChild(dialog);
            if (overlay.parentElement) overlay.parentElement.removeChild(overlay);
            this._activeDialogs = this._activeDialogs.filter(d => d !== dialog && d !== overlay);
        };

        overlay.addEventListener("click", closeDialog);

        // Title
        const titleEl = document.createElement("h3");
        titleEl.textContent = isEdit ? "Edit Macro" : "New Macro";
        titleEl.style.cssText = "margin: 0 0 12px 0; color: #e0e0e0;";
        dialog.appendChild(titleEl);

        // Name input
        const nameLabel = document.createElement("label");
        nameLabel.textContent = "Name:";
        nameLabel.style.cssText = "display: block; margin-bottom: 4px; color: #999;";
        dialog.appendChild(nameLabel);

        const nameInput = document.createElement("input");
        nameInput.type = "text";
        nameInput.placeholder = "e.g., Stage 1: Low Res";
        nameInput.value = isEdit ? existingMacro.name : "";
        nameInput.style.cssText = `
            width: 100%; padding: 8px; margin-bottom: 12px; background: #2a2a2a;
            border: 1px solid #444; border-radius: 4px; color: #e0e0e0; box-sizing: border-box;
        `;
        dialog.appendChild(nameInput);

        // Group checklist
        const checkLabel = document.createElement("label");
        checkLabel.textContent = "Select groups:";
        checkLabel.style.cssText = "display: block; margin-bottom: 4px; color: #999;";
        dialog.appendChild(checkLabel);

        const checkContainer = document.createElement("div");
        checkContainer.style.cssText = `
            max-height: 250px; overflow-y: auto; border: 1px solid #333; border-radius: 4px;
            padding: 4px; margin-bottom: 12px; background: #222;
        `;

        // Deduplicate group titles for the checklist
        const seenTitles = new Set();
        const selectedTitles = new Set(isEdit ? existingMacro.groupTitles : []);
        const checkboxes = [];

        // Also include stale titles from the existing macro so user can see them
        if (isEdit) {
            for (const t of existingMacro.groupTitles) {
                if (!groups.some(g => (g.title || "Untitled Group") === t)) {
                    // Stale title — show as disabled
                    const staleRow = document.createElement("div");
                    staleRow.style.cssText = "display: flex; align-items: center; padding: 4px 6px; color: #777;";
                    const staleCb = document.createElement("input");
                    staleCb.type = "checkbox";
                    staleCb.checked = true;
                    staleCb.disabled = true;
                    staleCb.style.marginRight = "8px";
                    const staleLabel2 = document.createElement("span");
                    staleLabel2.textContent = t + " (missing)";
                    staleLabel2.style.textDecoration = "line-through";
                    staleRow.appendChild(staleCb);
                    staleRow.appendChild(staleLabel2);
                    checkContainer.appendChild(staleRow);
                }
            }
        }

        for (const group of groups) {
            const title = group.title || "Untitled Group";
            if (seenTitles.has(title)) continue;
            seenTitles.add(title);

            const checkRow = document.createElement("div");
            checkRow.style.cssText = `
                display: flex; align-items: center; padding: 4px 6px; cursor: pointer;
                border-radius: 3px; transition: background 0.1s;
            `;
            checkRow.addEventListener("mouseenter", () => checkRow.style.background = "#333");
            checkRow.addEventListener("mouseleave", () => checkRow.style.background = "transparent");

            const cb = document.createElement("input");
            cb.type = "checkbox";
            cb.checked = selectedTitles.has(title);
            cb.style.marginRight = "8px";
            cb.style.cursor = "pointer";
            cb.dataset.title = title;
            checkboxes.push(cb);

            if (group.color) {
                const swatch = document.createElement("div");
                swatch.style.cssText = `
                    width: 8px; height: 8px; border-radius: 50%; background: ${group.color};
                    margin-right: 6px; flex-shrink: 0;
                `;
                checkRow.appendChild(swatch);
            }

            const cbLabel = document.createElement("span");
            cbLabel.textContent = title;
            cbLabel.style.cssText = "flex: 1; cursor: pointer;";

            // Show which macros this group belongs to
            if (this.macroManager) {
                const otherMacros = this.macroManager.macros.filter(m =>
                    m.groupTitles.includes(title) && (!isEdit || m.id !== existingMacro.id)
                );
                if (otherMacros.length > 0) {
                    const inLabel = document.createElement("span");
                    inLabel.textContent = `(in: ${otherMacros.map(m => m.name).join(", ")})`;
                    inLabel.style.cssText = "font-size: 10px; color: #666; margin-left: 4px;";
                    checkRow.appendChild(inLabel);
                }
            }

            checkRow.addEventListener("click", (e) => {
                if (e.target !== cb) cb.checked = !cb.checked;
            });

            checkRow.appendChild(cb);
            checkRow.appendChild(cbLabel);
            checkContainer.appendChild(checkRow);
        }

        dialog.appendChild(checkContainer);

        // Buttons
        const buttons = document.createElement("div");
        buttons.style.cssText = "display: flex; gap: 8px; justify-content: space-between;";

        const leftBtns = document.createElement("div");
        const rightBtns = document.createElement("div");
        rightBtns.style.cssText = "display: flex; gap: 8px;";

        // Delete button (edit mode only)
        if (isEdit) {
            const deleteBtn = document.createElement("button");
            deleteBtn.textContent = "Delete";
            deleteBtn.style.cssText = `
                padding: 8px 16px; background: #533; border: none; border-radius: 4px;
                color: #e0e0e0; cursor: pointer;
            `;
            let deleteConfirm = false;
            deleteBtn.addEventListener("click", () => {
                if (!deleteConfirm) {
                    deleteConfirm = true;
                    deleteBtn.textContent = "Click again to confirm";
                    deleteBtn.style.background = "#a33";
                    setTimeout(() => {
                        deleteConfirm = false;
                        deleteBtn.textContent = "Delete";
                        deleteBtn.style.background = "#533";
                    }, 2000);
                } else {
                    this.macroManager.deleteMacro(existingMacro.id);
                    closeDialog();
                    this._lastStateHash = "";
                    this.refreshGroups();
                }
            });
            leftBtns.appendChild(deleteBtn);
        }

        const cancelBtn = document.createElement("button");
        cancelBtn.textContent = "Cancel";
        cancelBtn.style.cssText = `
            padding: 8px 16px; background: #333; border: none; border-radius: 4px;
            color: #e0e0e0; cursor: pointer;
        `;
        cancelBtn.addEventListener("click", closeDialog);

        const saveBtn = document.createElement("button");
        saveBtn.textContent = isEdit ? "Save" : "Create";
        saveBtn.style.cssText = `
            padding: 8px 16px; background: #5a5; border: none; border-radius: 4px;
            color: white; cursor: pointer;
        `;
        saveBtn.addEventListener("click", () => {
            const name = nameInput.value.trim();
            if (!name) {
                nameInput.style.borderColor = "#a55";
                return;
            }
            const selected = checkboxes.filter(cb => cb.checked).map(cb => cb.dataset.title);
            if (selected.length === 0) {
                checkContainer.style.borderColor = "#a55";
                return;
            }

            if (isEdit) {
                this.macroManager.updateMacro(existingMacro.id, { name, groupTitles: selected });
            } else {
                this.macroManager.createMacro(name, selected);
            }

            closeDialog();
            this._lastStateHash = "";
            this.refreshGroups();
        });

        rightBtns.appendChild(cancelBtn);
        rightBtns.appendChild(saveBtn);
        buttons.appendChild(leftBtns);
        buttons.appendChild(rightBtns);
        dialog.appendChild(buttons);

        // Track dialogs for cleanup (P1)
        this._activeDialogs.push(dialog, overlay);

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        nameInput.focus();
    }

    // ─── Dialog cleanup helper ─────────────────────────────────────────

    _closeAllDialogs() {
        for (const el of this._activeDialogs) {
            if (el.parentElement) el.parentElement.removeChild(el);
        }
        this._activeDialogs = [];
        this._dialogOpen = false;
    }

    refreshPatterns() {
        this.patternsList.innerHTML = "";

        if (this.customPatterns.length === 0) {
            const empty = document.createElement("div");
            empty.textContent = "No custom patterns";
            empty.style.cssText = "padding: 8px; color: #666; font-style: italic; text-align: center;";
            this.patternsList.appendChild(empty);
            return;
        }

        for (const pattern of this.customPatterns) {
            const matchingNodes = this.findNodesByPattern(pattern.pattern);
            const hasActiveNodes = matchingNodes.some(n => n.mode === MODE_ALWAYS);

            const row = this.createToggleRow(
                `${pattern.name} (${matchingNodes.length})`,
                hasActiveNodes,
                (enable) => this.togglePattern(pattern, enable),
                null,
                () => this.removePattern(pattern)
            );

            this.patternsList.appendChild(row);
        }
    }

    toggleGroup(group, enable) {
        if (group.recomputeInsideNodes) {
            group.recomputeInsideNodes();
        }

        if (!group._nodes || group._nodes.length === 0) {
            console.warn("[FloatingPanel] No nodes in group:", group.title);
            return;
        }

        const newMode = enable ? MODE_ALWAYS : MODE_BYPASS;
        console.log(`[FloatingPanel] Setting ${group._nodes.length} nodes in "${group.title}" to mode ${newMode}`);

        if (app.graph.beforeChange) app.graph.beforeChange();
        for (const node of group._nodes) {
            node.mode = newMode;
        }
        if (app.graph.afterChange) app.graph.afterChange();

        app.graph.setDirtyCanvas(true, false);
        this._lastStateHash = ""; // force refresh
        this.refreshGroups();
    }

    navigateToGroup(group) {
        const canvas = app.canvas;
        canvas.centerOnNode(group);

        const zoomCurrent = canvas.ds?.scale || 1;
        const zoomX = canvas.canvas.width / group._size[0] - 0.02;
        const zoomY = canvas.canvas.height / group._size[1] - 0.02;
        canvas.setZoom(Math.min(zoomCurrent, zoomX, zoomY), [
            canvas.canvas.width / 2,
            canvas.canvas.height / 2
        ]);
        canvas.setDirty(true, true);
    }

    setAllGroupsMode(mode) {
        const groups = app.graph._groups || [];
        if (app.graph.beforeChange) app.graph.beforeChange();
        for (const group of groups) {
            if (group.recomputeInsideNodes) {
                group.recomputeInsideNodes();
            }
            for (const node of group._nodes || []) {
                node.mode = mode;
            }
        }
        if (app.graph.afterChange) app.graph.afterChange();
        app.graph.setDirtyCanvas(true, false);
        this._lastStateHash = "";
        this.refreshGroups();
    }

    // Pattern matching (similar to NodeBypasser)
    findNodesByPattern(pattern) {
        if (!app.graph) return [];

        const nodes = app.graph._nodes || [];

        if (pattern.startsWith('@')) {
            const groupName = pattern.substring(1);
            return this.findNodesInGroup(groupName);
        }

        if (this.isRegexPattern(pattern)) {
            return this.findNodesByRegex(pattern, nodes);
        }

        return nodes.filter(node =>
            node.type.toLowerCase().includes(pattern.toLowerCase()) ||
            (node.title && node.title.toLowerCase().includes(pattern.toLowerCase()))
        );
    }

    isRegexPattern(pattern) {
        return pattern.includes('*') ||
               pattern.includes('!') ||
               pattern.includes('^') ||
               pattern.includes('$') ||
               pattern.includes('[') ||
               pattern.includes(']');
    }

    findNodesByRegex(pattern, nodes) {
        try {
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

            return nodes.filter(node => {
                const matches = regex.test(node.type) ||
                               (node.title && regex.test(node.title));
                return isExclusion ? !matches : matches;
            });
        } catch (e) {
            console.error("[FloatingPanel] Regex error:", e);
            return [];
        }
    }

    findNodesInGroup(groupName) {
        const groups = app.graph._groups || [];
        const matchingGroups = groups.filter(g =>
            g.title && g.title.toLowerCase().includes(groupName.toLowerCase())
        );

        const result = [];
        const seenIds = new Set();

        for (const group of matchingGroups) {
            if (group.recomputeInsideNodes) {
                group.recomputeInsideNodes();
            }
            for (const node of group._nodes || []) {
                if (!seenIds.has(node.id)) {
                    result.push(node);
                    seenIds.add(node.id);
                }
            }
        }

        return result;
    }

    togglePattern(pattern, enable) {
        const nodes = this.findNodesByPattern(pattern.pattern);
        const newMode = enable ? MODE_ALWAYS : (pattern.mode === 'bypass' ? MODE_BYPASS : MODE_NEVER);

        if (app.graph.beforeChange) app.graph.beforeChange();
        for (const node of nodes) {
            node.mode = newMode;
        }
        if (app.graph.afterChange) app.graph.afterChange();

        app.graph.setDirtyCanvas(true, false);
        this.refreshPatterns();
    }

    showAddPatternDialog() {
        this._dialogOpen = true; // P0: pause polling

        const dialog = document.createElement("div");
        dialog.style.cssText = `
            position: fixed;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            background: #1a1a1a;
            border: 1px solid #444;
            border-radius: 8px;
            padding: 16px;
            z-index: 100003;
            min-width: 300px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
        `;

        const title = document.createElement("h3");
        title.textContent = "Add Custom Pattern";
        title.style.cssText = "margin: 0 0 12px 0; color: #e0e0e0;";

        const nameLabel = document.createElement("label");
        nameLabel.textContent = "Name:";
        nameLabel.style.cssText = "display: block; margin-bottom: 4px; color: #999;";

        const nameInput = document.createElement("input");
        nameInput.type = "text";
        nameInput.placeholder = "e.g., My Samplers";
        nameInput.style.cssText = `
            width: 100%;
            padding: 8px;
            margin-bottom: 12px;
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 4px;
            color: #e0e0e0;
            box-sizing: border-box;
        `;

        const patternLabel = document.createElement("label");
        patternLabel.textContent = "Pattern:";
        patternLabel.style.cssText = "display: block; margin-bottom: 4px; color: #999;";

        const patternInput = document.createElement("input");
        patternInput.type = "text";
        patternInput.placeholder = "e.g., KSampler*, @GroupName, LoadImage";
        patternInput.style.cssText = nameInput.style.cssText;

        // P0: XSS safe — use textContent, not innerHTML for help text
        const helpText = document.createElement("div");
        helpText.style.cssText = "margin-bottom: 12px; font-size: 11px; color: #666;";
        helpText.textContent = "Patterns: * = wildcard, @GroupName = group, ! = exclude";

        const modeLabel = document.createElement("label");
        modeLabel.textContent = "Off Mode:";
        modeLabel.style.cssText = "display: block; margin-bottom: 4px; color: #999;";

        const modeSelect = document.createElement("select");
        modeSelect.style.cssText = `
            width: 100%;
            padding: 8px;
            margin-bottom: 16px;
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 4px;
            color: #e0e0e0;
        `;
        const optMute = document.createElement("option");
        optMute.value = "mute";
        optMute.textContent = "Mute (completely disabled)";
        const optBypass = document.createElement("option");
        optBypass.value = "bypass";
        optBypass.textContent = "Bypass (pass-through)";
        modeSelect.appendChild(optMute);
        modeSelect.appendChild(optBypass);

        const buttons = document.createElement("div");
        buttons.style.cssText = "display: flex; gap: 8px; justify-content: flex-end;";

        const closeDialog = () => {
            this._dialogOpen = false;
            if (dialog.parentElement) document.body.removeChild(dialog);
            if (overlay.parentElement) document.body.removeChild(overlay);
            this._activeDialogs = this._activeDialogs.filter(d => d !== dialog && d !== overlay);
        };

        const cancelBtn = document.createElement("button");
        cancelBtn.textContent = "Cancel";
        cancelBtn.style.cssText = `
            padding: 8px 16px;
            background: #333;
            border: none;
            border-radius: 4px;
            color: #e0e0e0;
            cursor: pointer;
        `;
        cancelBtn.addEventListener("click", closeDialog);

        const addBtn = document.createElement("button");
        addBtn.textContent = "Add";
        addBtn.style.cssText = `
            padding: 8px 16px;
            background: #5a5;
            border: none;
            border-radius: 4px;
            color: white;
            cursor: pointer;
        `;
        addBtn.addEventListener("click", () => {
            const name = nameInput.value.trim();
            const pattern = patternInput.value.trim();

            if (name && pattern) {
                this.customPatterns.push({
                    name,
                    pattern,
                    mode: modeSelect.value
                });
                this.saveState();
                this.refreshPatterns();
            }

            closeDialog();
        });

        buttons.appendChild(cancelBtn);
        buttons.appendChild(addBtn);

        dialog.appendChild(title);
        dialog.appendChild(nameLabel);
        dialog.appendChild(nameInput);
        dialog.appendChild(patternLabel);
        dialog.appendChild(patternInput);
        dialog.appendChild(helpText);
        dialog.appendChild(modeLabel);
        dialog.appendChild(modeSelect);
        dialog.appendChild(buttons);

        // Overlay
        const overlay = document.createElement("div");
        overlay.style.cssText = `
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.5);
            z-index: 100002;
        `;
        overlay.addEventListener("click", closeDialog);

        // Track for cleanup (P1)
        this._activeDialogs.push(dialog, overlay);

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        nameInput.focus();
    }

    removePattern(pattern) {
        const index = this.customPatterns.indexOf(pattern);
        if (index > -1) {
            this.customPatterns.splice(index, 1);
            this.saveState();
            this.refreshPatterns();
        }
    }

    startRefreshLoop() {
        this.refreshInterval = setInterval(() => {
            // P0: pause during dialogs, skip if not visible
            if (this._dialogOpen) return;
            if (this.isVisible && !this.isCollapsed && app.graph) {
                // P0: try/catch prevents one crash from killing the loop
                try {
                    this.refreshGroups();
                    this.refreshPatterns();
                } catch (e) {
                    console.error("[FloatingPanel] Refresh error:", e);
                }
            }
        }, 500);
    }

    destroy() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        this._closeAllDialogs();
        if (this.container && this.container.parentElement) {
            this.container.parentElement.removeChild(this.container);
        }
    }
}

// Global instance
let floatingPanelInstance = null;

// Public API
window.NVFloatingPanel = {
    show: () => floatingPanelInstance?.show(),
    hide: () => floatingPanelInstance?.hide(),
    toggle: () => floatingPanelInstance?.toggle(),
    getInstance: () => floatingPanelInstance
};

// Register extension
app.registerExtension({
    name: "NV_Comfy_Utils.FloatingPanel",

    async setup() {
        console.log("[FloatingPanel] Setting up...");

        // Wait for app to be ready, with timeout to avoid infinite hang
        if (!app.graph) {
            await new Promise((resolve) => {
                let elapsed = 0;
                const check = setInterval(() => {
                    elapsed += 100;
                    if (app.graph) {
                        clearInterval(check);
                        resolve();
                    } else if (elapsed > 15000) {
                        clearInterval(check);
                        console.warn("[FloatingPanel] Timed out waiting for app.graph after 15s, proceeding anyway");
                        resolve();
                    }
                }, 100);
            });
        }

        // Create the panel
        try {
            floatingPanelInstance = new FloatingPanel();
            console.log("[FloatingPanel] Panel created, visible:", floatingPanelInstance.isVisible);
        } catch (e) {
            console.error("[FloatingPanel] Failed to create panel:", e);
            return;
        }

        // Hook workflow/tab switching — app.graph is replaced on load,
        // so we invalidate the dirty-check hash and force a refresh.
        const origLoadGraphData = app.loadGraphData;
        app.loadGraphData = async function() {
            const result = await origLoadGraphData.apply(this, arguments);
            if (floatingPanelInstance) {
                floatingPanelInstance._lastStateHash = "";
                floatingPanelInstance.refreshGroups();
                floatingPanelInstance.refreshPatterns();
            }
            return result;
        };

        // Add keyboard shortcut (Ctrl+Shift+P to toggle)
        document.addEventListener("keydown", (e) => {
            if (e.ctrlKey && e.shiftKey && e.key === "P") {
                e.preventDefault();
                floatingPanelInstance.toggle();
            }
        });

        // Add sidebar button to ComfyUI menu bar
        try {
            const { ComfyButton } = await import("../../scripts/ui/components/button.js");

            if (!ComfyButton) {
                console.warn("[FloatingPanel] ComfyButton is undefined (deprecated API removed)");
            } else {
                const toggleBtn = new ComfyButton({
                    icon: "toggle-switch",
                    action: () => floatingPanelInstance.toggle(),
                    tooltip: "Quick Toggle (Ctrl+Shift+P)",
                    content: "Toggle",
                    classList: "comfyui-button comfyui-menu-mobile-collapse"
                });

                if (app.menu?.settingsGroup?.element) {
                    app.menu.settingsGroup.element.before(toggleBtn.element);
                    console.log("[FloatingPanel] Sidebar button added via settingsGroup");
                } else if (app.menu?.element) {
                    app.menu.element.appendChild(toggleBtn.element);
                    console.log("[FloatingPanel] Sidebar button added via menu.element fallback");
                } else {
                    console.warn("[FloatingPanel] No menu anchor found - button not added. Use Ctrl+Shift+P to toggle.");
                }
            }
        } catch (e) {
            console.warn("[FloatingPanel] Could not add sidebar button:", e);
        }

        console.log("[FloatingPanel] Ready! Toggle with sidebar button or Ctrl+Shift+P");
    }
});
