/**
 * Variables Panel — Console-driven variable management with drag-and-drop
 *
 * Inspired by Unreal Engine's "My Blueprint" panel. This is the single
 * source of truth for variable CRUD operations. Variables are created,
 * deleted, and renamed here. Getter nodes are placed via drag-and-drop
 * from the panel onto the canvas.
 *
 * Features:
 * - Create / delete / rename variables
 * - Type-colored indicators (UE Blueprint style)
 * - Drag variable rows to canvas → creates GetVariableNode
 * - Drag to node input slot → creates getter + auto-connects
 * - Source info display (which node feeds each variable)
 * - Orphan detection (getters with no setter)
 * - Search/filter bar
 * - Draggable, collapsible, state persisted to localStorage
 */

import { app } from "../../scripts/app.js";
import { variableManager } from "./variable_manager.js";

const STORAGE_KEY = "NV_VariablesPanel_State";
const DEFAULT_POSITION = { x: 320, y: 100 };

class VariablesPanel {
    constructor() {
        this.container = null;
        this.header = null;
        this.content = null;
        this.searchInput = null;
        this.variablesSection = null;
        this.orphansSection = null;
        this.isCollapsed = false;
        this.isDragging = false;
        this.dragOffset = { x: 0, y: 0 };
        this.position = { ...DEFAULT_POSITION };
        this.refreshInterval = null;
        this.isVisible = false;
        this._lastHash = "";
        this.variableMap = new Map();
        this.expandedVars = new Set();

        // Drag-and-drop state
        this._varDrag = {
            active: false,
            varName: null,
            ghost: null,
        };

        this.loadState();
        this.createPanel();
        this.startRefreshLoop();
    }

    // ===== State Persistence =====

    loadState() {
        try {
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved) {
                const state = JSON.parse(saved);
                this.position = state.position || { ...DEFAULT_POSITION };
                this.isCollapsed = state.isCollapsed || false;
                this.isVisible = state.isVisible || false;
            }
        } catch (e) {
            console.warn("[VariablesPanel] Failed to load state:", e);
        }
    }

    saveState() {
        try {
            const state = {
                position: this.position,
                isCollapsed: this.isCollapsed,
                isVisible: this.isVisible,
            };
            localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        } catch (e) {
            console.warn("[VariablesPanel] Failed to save state:", e);
        }
    }

    // ===== DOM Construction =====

    createPanel() {
        // Main container
        this.container = document.createElement("div");
        this.container.id = "nv-variables-panel";
        this.container.style.cssText = `
            position: fixed;
            left: ${this.position.x}px;
            top: ${this.position.y}px;
            width: 300px;
            background: #1a1a1a;
            border: 1px solid #444;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            z-index: 10000;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 12px;
            color: #e0e0e0;
            user-select: none;
            display: ${this.isVisible ? 'block' : 'none'};
        `;

        // Header
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
        title.textContent = "Variables";
        title.style.fontWeight = "600";

        const headerButtons = document.createElement("div");
        headerButtons.style.cssText = "display: flex; gap: 4px;";

        // Create Variable button (+)
        const addBtn = this._createIconButton("+", "Create variable", () => this._showCreateInput());
        addBtn.style.color = "#4ade80";
        addBtn.style.fontWeight = "bold";
        addBtn.style.fontSize = "16px";

        const refreshBtn = this._createIconButton("\u21BB", "Refresh", () => {
            this._lastHash = "";
            this.refresh();
        });
        this.collapseBtn = this._createIconButton(this.isCollapsed ? "\u25BC" : "\u25B2", "Toggle collapse", () => this.toggleCollapse());
        const closeBtn = this._createIconButton("\u00D7", "Hide panel", () => this.hide());
        closeBtn.style.fontSize = "16px";

        headerButtons.appendChild(addBtn);
        headerButtons.appendChild(refreshBtn);
        headerButtons.appendChild(this.collapseBtn);
        headerButtons.appendChild(closeBtn);

        this.header.appendChild(title);
        this.header.appendChild(headerButtons);

        // Content area
        this.content = document.createElement("div");
        this.content.style.cssText = `
            max-height: 500px;
            overflow-y: auto;
            display: ${this.isCollapsed ? 'none' : 'block'};
        `;

        // Create variable inline input (hidden by default)
        this.createInputContainer = document.createElement("div");
        this.createInputContainer.style.cssText = "padding: 6px 8px; display: none;";

        this.createInput = document.createElement("input");
        this.createInput.type = "text";
        this.createInput.placeholder = "New variable name...";
        this.createInput.style.cssText = `
            width: 100%;
            padding: 6px 8px;
            background: #2a2a2a;
            border: 1px solid #4ade80;
            border-radius: 4px;
            color: #e0e0e0;
            box-sizing: border-box;
            font-size: 12px;
            outline: none;
        `;
        this.createInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                this._handleCreateVariable();
            } else if (e.key === "Escape") {
                this._hideCreateInput();
            }
            e.stopPropagation();
        });
        this.createInput.addEventListener("blur", () => {
            // Small delay so click events can fire first
            setTimeout(() => this._hideCreateInput(), 150);
        });

        this.createError = document.createElement("div");
        this.createError.style.cssText = "color: #f44; font-size: 10px; min-height: 14px; padding: 2px 0 0 0;";

        this.createInputContainer.appendChild(this.createInput);
        this.createInputContainer.appendChild(this.createError);

        // Search bar
        const searchBar = document.createElement("div");
        searchBar.style.cssText = "padding: 6px 8px;";

        this.searchInput = document.createElement("input");
        this.searchInput.type = "text";
        this.searchInput.placeholder = "Filter variables...";
        this.searchInput.style.cssText = `
            width: 100%;
            padding: 6px 8px;
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 4px;
            color: #e0e0e0;
            box-sizing: border-box;
            font-size: 12px;
            outline: none;
        `;
        this.searchInput.addEventListener("input", () => this._applyFilter(this.searchInput.value));
        this.searchInput.addEventListener("focus", () => this.searchInput.style.borderColor = "#5af");
        this.searchInput.addEventListener("blur", () => this.searchInput.style.borderColor = "#444");
        this.searchInput.addEventListener("keydown", (e) => e.stopPropagation());
        searchBar.appendChild(this.searchInput);

        // Sections
        this.variablesSection = document.createElement("div");
        this.orphansSection = document.createElement("div");

        this.content.appendChild(this.createInputContainer);
        this.content.appendChild(searchBar);
        this.content.appendChild(this.variablesSection);
        this.content.appendChild(this.orphansSection);

        this.container.appendChild(this.header);
        this.container.appendChild(this.content);

        document.body.appendChild(this.container);

        this._setupPanelDrag();
        this._setupVarDragListeners();
        this.refresh();
    }

    // ===== Create Variable UI =====

    _showCreateInput() {
        this.createInputContainer.style.display = "block";
        this.createInput.value = "";
        this.createError.textContent = "";
        this.createInput.focus();
    }

    _hideCreateInput() {
        this.createInputContainer.style.display = "none";
        this.createInput.value = "";
        this.createError.textContent = "";
    }

    _handleCreateVariable() {
        const name = this.createInput.value.trim();
        if (!name) {
            this.createError.textContent = "Name cannot be empty";
            return;
        }

        const existingNames = variableManager.getVariableNames();
        if (existingNames.includes(name)) {
            this.createError.textContent = `"${name}" already exists`;
            return;
        }

        const setter = variableManager.createVariable(name);
        if (setter) {
            this._hideCreateInput();
            this._lastHash = "";
            this.refresh();
        } else {
            this.createError.textContent = "Failed to create variable";
        }
    }

    // ===== Button Helpers =====

    _createIconButton(icon, tooltip, onClick) {
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

    _createActionButton(text, color, onClick) {
        const btn = document.createElement("button");
        btn.textContent = text;
        btn.style.cssText = `
            padding: 4px 8px;
            background: ${color};
            border: none;
            border-radius: 4px;
            color: white;
            cursor: pointer;
            font-size: 10px;
            font-weight: 500;
            transition: filter 0.15s;
        `;
        btn.addEventListener("mouseenter", () => btn.style.filter = "brightness(1.2)");
        btn.addEventListener("mouseleave", () => btn.style.filter = "none");
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            onClick();
        });
        return btn;
    }

    _createSectionHeader(title, count, warningLevel = 0) {
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

        const label = document.createElement("span");
        label.textContent = `${title} (${count})`;
        header.appendChild(label);

        if (warningLevel > 0) {
            const badge = document.createElement("span");
            badge.textContent = warningLevel === 1 ? "!" : "!!";
            badge.style.cssText = `
                color: ${warningLevel === 1 ? '#a33' : '#a83'};
                font-weight: bold;
                font-size: 13px;
            `;
            header.appendChild(badge);
        }

        return header;
    }

    // ===== Rendering =====

    refresh() {
        if (!app.graph) return;

        this.variableMap = variableManager.getAllVariables();
        this._renderVariablesSection();
        this._renderOrphansSection();

        if (this.searchInput && this.searchInput.value) {
            this._applyFilter(this.searchInput.value);
        }
    }

    _renderVariablesSection() {
        this.variablesSection.innerHTML = "";

        // Filter to healthy variables (have a setter, no duplicates)
        const healthy = [];
        for (const [name, info] of this.variableMap) {
            if (!info.hasOrphanGetters && !info.hasDuplicateSetters) {
                healthy.push(info);
            }
            // Also include duplicates as "variables" since they still have setters
            if (info.hasDuplicateSetters && !info.hasOrphanGetters) {
                healthy.push(info);
            }
        }

        // Deduplicate (in case both conditions matched)
        const seen = new Set();
        const unique = healthy.filter(info => {
            if (seen.has(info.name)) return false;
            seen.add(info.name);
            return true;
        });

        if (unique.length === 0 && this.variableMap.size === 0) {
            const header = this._createSectionHeader("Variables", 0);
            const empty = document.createElement("div");
            empty.textContent = "No variables yet. Click + to create one.";
            empty.style.cssText = "padding: 12px; color: #666; font-style: italic; text-align: center;";
            this.variablesSection.appendChild(header);
            this.variablesSection.appendChild(empty);
            return;
        }

        const header = this._createSectionHeader("Variables", unique.length);
        this.variablesSection.appendChild(header);

        for (const info of unique) {
            const row = this._createVariableRow(info);
            this.variablesSection.appendChild(row);
        }
    }

    _renderOrphansSection() {
        this.orphansSection.innerHTML = "";

        const orphans = [];
        for (const [name, info] of this.variableMap) {
            if (info.hasOrphanGetters) {
                orphans.push(info);
            }
        }

        if (orphans.length === 0) return;

        const header = this._createSectionHeader("Orphans", orphans.length, 1);
        this.orphansSection.appendChild(header);

        for (const info of orphans) {
            for (const getter of info.getters) {
                const row = this._createOrphanRow(info.name, getter);
                this.orphansSection.appendChild(row);
            }
        }
    }

    _createVariableRow(info) {
        const container = document.createElement("div");
        container.classList.add("nv-var-row");
        container.dataset.varName = info.name.toLowerCase();
        container.style.cssText = `
            margin: 2px 4px;
            border-radius: 4px;
            background: #2a2a2a;
            overflow: hidden;
        `;

        const isExpanded = this.expandedVars.has(info.name);

        // --- Main row (draggable + clickable) ---
        const mainRow = document.createElement("div");
        mainRow.style.cssText = `
            display: flex;
            align-items: center;
            padding: 6px 8px;
            cursor: grab;
            transition: background 0.15s;
        `;
        mainRow.addEventListener("mouseenter", () => {
            if (!this._varDrag.active) mainRow.style.background = "#333";
        });
        mainRow.addEventListener("mouseleave", () => {
            if (!this._varDrag.active) mainRow.style.background = "transparent";
        });

        // Type-colored dot (UE Blueprint style)
        const typeColor = variableManager.getTypeColor(info.type);
        const dot = document.createElement("span");
        dot.style.cssText = `
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
            flex-shrink: 0;
            ${info.isConnected
                ? `background: ${typeColor};`
                : `border: 2px solid ${typeColor}; box-sizing: border-box;`
            }
        `;

        // Variable name
        const nameSpan = document.createElement("span");
        nameSpan.textContent = info.name;
        nameSpan.style.cssText = `
            flex: 1;
            font-weight: 500;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        `;

        // Type badge
        const typeBadge = document.createElement("span");
        typeBadge.textContent = info.type === "unconnected" ? "\u2014" : info.type;
        typeBadge.style.cssText = `
            font-size: 10px;
            color: ${typeColor};
            margin-right: 4px;
            flex-shrink: 0;
        `;

        // Context menu button
        const menuBtn = document.createElement("button");
        menuBtn.textContent = "\u22EE";
        menuBtn.title = "Actions";
        menuBtn.style.cssText = `
            background: transparent;
            border: none;
            color: #666;
            cursor: pointer;
            padding: 0 4px;
            font-size: 16px;
            line-height: 1;
            border-radius: 3px;
            flex-shrink: 0;
        `;
        menuBtn.addEventListener("mouseenter", () => menuBtn.style.color = "#aaa");
        menuBtn.addEventListener("mouseleave", () => menuBtn.style.color = "#666");
        menuBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            this._showRowContextMenu(info, e);
        });

        mainRow.appendChild(dot);
        mainRow.appendChild(nameSpan);
        mainRow.appendChild(typeBadge);
        mainRow.appendChild(menuBtn);

        // --- Info line ---
        const infoLine = document.createElement("div");
        infoLine.style.cssText = `
            padding: 0 8px 4px 26px;
            font-size: 10px;
            color: #888;
        `;

        const getterCount = info.getters.length;
        const sourceInfo = variableManager.getSourceInfo(info.name);
        const sourceText = sourceInfo
            ? `${sourceInfo.nodeTitle} \u2192 ${sourceInfo.outputName}`
            : "no source";
        infoLine.textContent = `${sourceText} \u00B7 ${getterCount} getter${getterCount !== 1 ? 's' : ''}`;

        // --- Expandable details ---
        const details = document.createElement("div");
        details.style.cssText = `
            display: ${isExpanded ? 'block' : 'none'};
            padding: 0 4px 4px 4px;
            border-top: 1px solid #333;
        `;

        // Chevron for expand/collapse
        const chevron = document.createElement("span");
        chevron.textContent = isExpanded ? "\u25BE" : "\u25B8";
        chevron.style.cssText = "margin-right: 4px; font-size: 10px; color: #888; cursor: pointer; flex-shrink: 0;";
        chevron.addEventListener("click", (e) => {
            e.stopPropagation();
            if (this.expandedVars.has(info.name)) {
                this.expandedVars.delete(info.name);
                details.style.display = "none";
                chevron.textContent = "\u25B8";
            } else {
                this.expandedVars.add(info.name);
                details.style.display = "block";
                chevron.textContent = "\u25BE";
            }
        });

        // Insert chevron at the start of mainRow (before dot)
        mainRow.insertBefore(chevron, dot);

        // Source sub-row
        if (sourceInfo) {
            const sourceRow = this._createSourceSubRow(sourceInfo);
            details.appendChild(sourceRow);
        }

        // Getter sub-rows
        for (const getter of info.getters) {
            details.appendChild(this._createNodeSubRow(getter, "GET"));
        }

        // Duplicate setter warning
        if (info.hasDuplicateSetters) {
            const warnRow = document.createElement("div");
            warnRow.style.cssText = "padding: 4px 8px; font-size: 10px; color: #a83;";
            warnRow.textContent = `!! ${info.allSetters.length} setters (duplicate)`;
            details.appendChild(warnRow);
        }

        // Action buttons
        const actions = document.createElement("div");
        actions.style.cssText = "display: flex; gap: 4px; padding: 4px 8px;";
        actions.appendChild(this._createActionButton("Select All", "#445", () => this._selectAllForVariable(info)));
        actions.appendChild(this._createActionButton("Rename", "#454", () => this._showRenameDialog(info)));
        details.appendChild(actions);

        // --- Drag-and-drop from main row ---
        mainRow.addEventListener("mousedown", (e) => {
            // Only start drag from the row itself, not from buttons
            if (e.target.tagName === "BUTTON" || e.target === chevron) return;
            e.preventDefault();
            this._startVarDrag(info.name, info.type, e);
        });

        container.appendChild(mainRow);
        container.appendChild(infoLine);
        container.appendChild(details);

        return container;
    }

    _createSourceSubRow(sourceInfo) {
        const row = document.createElement("div");
        row.style.cssText = `
            display: flex;
            align-items: center;
            padding: 3px 8px;
            margin: 1px 0;
            border-radius: 3px;
            transition: background 0.15s;
        `;
        row.addEventListener("mouseenter", () => row.style.background = "#383838");
        row.addEventListener("mouseleave", () => row.style.background = "transparent");

        const roleSpan = document.createElement("span");
        roleSpan.textContent = "SRC";
        roleSpan.style.cssText = `
            color: #f59e0b;
            font-size: 10px;
            font-weight: 600;
            width: 28px;
            flex-shrink: 0;
        `;

        const label = document.createElement("span");
        label.textContent = `${sourceInfo.nodeTitle} \u2192 ${sourceInfo.outputName}`;
        label.style.cssText = `
            flex: 1;
            font-size: 11px;
            color: #ccc;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        `;

        const navBtn = this._createNavigateButton(sourceInfo.node);

        row.appendChild(roleSpan);
        row.appendChild(label);
        row.appendChild(navBtn);

        return row;
    }

    _createNodeSubRow(node, role) {
        const row = document.createElement("div");
        row.style.cssText = `
            display: flex;
            align-items: center;
            padding: 3px 8px;
            margin: 1px 0;
            border-radius: 3px;
            transition: background 0.15s;
        `;
        row.addEventListener("mouseenter", () => row.style.background = "#383838");
        row.addEventListener("mouseleave", () => row.style.background = "transparent");

        const roleSpan = document.createElement("span");
        const roleColor = role === "SET" ? "#5a5" : "#5af";
        roleSpan.textContent = role;
        roleSpan.style.cssText = `
            color: ${roleColor};
            font-size: 10px;
            font-weight: 600;
            width: 28px;
            flex-shrink: 0;
        `;

        const label = document.createElement("span");
        const displayName = node.title || `${role === "SET" ? "Set" : "Get"} Variable`;
        label.textContent = `${displayName} (id:${node.id})`;
        label.style.cssText = `
            flex: 1;
            font-size: 11px;
            color: #ccc;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        `;

        row.appendChild(roleSpan);
        row.appendChild(label);
        row.appendChild(this._createNavigateButton(node));

        return row;
    }

    _createOrphanRow(varName, getter) {
        const row = document.createElement("div");
        row.classList.add("nv-var-row");
        row.dataset.varName = varName.toLowerCase();
        row.style.cssText = `
            display: flex;
            align-items: center;
            padding: 6px 8px;
            margin: 2px 4px;
            background: rgba(170, 51, 51, 0.15);
            border-left: 3px solid #a33;
            border-radius: 4px;
            transition: background 0.15s;
        `;
        row.addEventListener("mouseenter", () => row.style.background = "rgba(170, 51, 51, 0.25)");
        row.addEventListener("mouseleave", () => row.style.background = "rgba(170, 51, 51, 0.15)");

        const icon = document.createElement("span");
        icon.textContent = "!";
        icon.style.cssText = "color: #a33; font-weight: bold; margin-right: 8px; flex-shrink: 0;";

        const label = document.createElement("span");
        const displayName = getter.title || "Get Variable";
        label.textContent = `"${varName}" \u2014 no setter`;
        label.title = `${displayName} (id:${getter.id})`;
        label.style.cssText = `
            flex: 1;
            font-size: 11px;
            color: #ccc;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        `;

        row.appendChild(icon);
        row.appendChild(label);
        row.appendChild(this._createNavigateButton(getter));

        return row;
    }

    _createNavigateButton(node) {
        const btn = document.createElement("button");
        btn.textContent = "\u2192";
        btn.title = "Go to node";
        btn.style.cssText = `
            background: transparent;
            border: none;
            color: #666;
            cursor: pointer;
            padding: 2px 6px;
            font-size: 12px;
            flex-shrink: 0;
        `;
        btn.addEventListener("mouseenter", () => btn.style.color = "#aaa");
        btn.addEventListener("mouseleave", () => btn.style.color = "#666");
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            this._navigateToNode(node);
        });
        return btn;
    }

    // ===== Row Context Menu =====

    _showRowContextMenu(info, event) {
        // Build a simple dropdown menu at click position
        const existing = document.getElementById("nv-var-context-menu");
        if (existing) existing.remove();

        const menu = document.createElement("div");
        menu.id = "nv-var-context-menu";
        menu.style.cssText = `
            position: fixed;
            left: ${event.clientX}px;
            top: ${event.clientY}px;
            background: #1a1a1a;
            border: 1px solid #444;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            z-index: 10002;
            min-width: 160px;
            padding: 4px 0;
        `;

        const items = [
            {
                label: "Go to Source",
                enabled: !!variableManager.getSourceInfo(info.name),
                action: () => {
                    const src = variableManager.getSourceInfo(info.name);
                    if (src) this._navigateToNode(src.node);
                }
            },
            {
                label: "Select All Getters",
                enabled: info.getters.length > 0,
                action: () => this._selectAllForVariable(info)
            },
            null, // separator
            {
                label: "Rename",
                enabled: true,
                action: () => this._showRenameDialog(info)
            },
            {
                label: "Unassign Source",
                enabled: !!variableManager.getSourceInfo(info.name),
                action: () => {
                    variableManager.unassignSource(info.name);
                    this._lastHash = "";
                    this.refresh();
                }
            },
            null, // separator
            {
                label: "Delete Variable",
                enabled: true,
                color: "#f44",
                action: () => this._confirmDelete(info)
            },
        ];

        for (const item of items) {
            if (item === null) {
                const sep = document.createElement("div");
                sep.style.cssText = "height: 1px; background: #333; margin: 4px 0;";
                menu.appendChild(sep);
                continue;
            }

            const row = document.createElement("div");
            row.textContent = item.label;
            row.style.cssText = `
                padding: 6px 12px;
                cursor: ${item.enabled ? 'pointer' : 'default'};
                color: ${!item.enabled ? '#555' : (item.color || '#e0e0e0')};
                font-size: 12px;
                transition: background 0.1s;
            `;
            if (item.enabled) {
                row.addEventListener("mouseenter", () => row.style.background = "#333");
                row.addEventListener("mouseleave", () => row.style.background = "transparent");
                row.addEventListener("click", () => {
                    menu.remove();
                    item.action();
                });
            }
            menu.appendChild(row);
        }

        document.body.appendChild(menu);

        // Close on click outside
        const closeMenu = (e) => {
            if (!menu.contains(e.target)) {
                menu.remove();
                document.removeEventListener("mousedown", closeMenu);
            }
        };
        setTimeout(() => document.addEventListener("mousedown", closeMenu), 0);
    }

    // ===== Delete Confirmation =====

    _confirmDelete(info) {
        const getterCount = info.getters.length;
        const msg = `Delete "${info.name}"?\n\nThis will remove the variable and ${getterCount} getter node${getterCount !== 1 ? 's' : ''} from the canvas.`;

        const overlay = document.createElement("div");
        overlay.style.cssText = `
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.5);
            z-index: 10000;
        `;

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
            z-index: 10001;
            min-width: 300px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
        `;

        const title = document.createElement("h3");
        title.textContent = "Delete Variable";
        title.style.cssText = "margin: 0 0 8px 0; color: #e0e0e0; font-size: 14px;";

        const body = document.createElement("div");
        body.textContent = `This will remove "${info.name}" and ${getterCount} getter node${getterCount !== 1 ? 's' : ''} from the canvas.`;
        body.style.cssText = "margin-bottom: 16px; font-size: 12px; color: #999;";

        const buttons = document.createElement("div");
        buttons.style.cssText = "display: flex; gap: 8px; justify-content: flex-end;";

        const cleanup = () => {
            document.body.removeChild(dialog);
            document.body.removeChild(overlay);
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
            font-size: 12px;
        `;
        cancelBtn.addEventListener("click", cleanup);

        const deleteBtn = document.createElement("button");
        deleteBtn.textContent = "Delete";
        deleteBtn.style.cssText = `
            padding: 8px 16px;
            background: #c53030;
            border: none;
            border-radius: 4px;
            color: white;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
        `;
        deleteBtn.addEventListener("click", () => {
            variableManager.deleteVariable(info.name);
            cleanup();
            this._lastHash = "";
            this.refresh();
        });

        overlay.addEventListener("click", cleanup);

        buttons.appendChild(cancelBtn);
        buttons.appendChild(deleteBtn);

        dialog.appendChild(title);
        dialog.appendChild(body);
        dialog.appendChild(buttons);

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
    }

    // ===== Rename Dialog =====

    _showRenameDialog(info) {
        const overlay = document.createElement("div");
        overlay.style.cssText = `
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.5);
            z-index: 10000;
        `;

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
            z-index: 10001;
            min-width: 300px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
        `;

        const title = document.createElement("h3");
        title.textContent = "Rename Variable";
        title.style.cssText = "margin: 0 0 12px 0; color: #e0e0e0; font-size: 14px;";

        const label = document.createElement("label");
        label.textContent = "New name:";
        label.style.cssText = "display: block; margin-bottom: 4px; color: #999; font-size: 12px;";

        const input = document.createElement("input");
        input.type = "text";
        input.value = info.name;
        input.style.cssText = `
            width: 100%;
            padding: 8px;
            margin-bottom: 4px;
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 4px;
            color: #e0e0e0;
            box-sizing: border-box;
            font-size: 12px;
            outline: none;
        `;

        const errorMsg = document.createElement("div");
        errorMsg.style.cssText = "color: #f44; font-size: 11px; min-height: 16px; margin-bottom: 8px;";

        const buttons = document.createElement("div");
        buttons.style.cssText = "display: flex; gap: 8px; justify-content: flex-end;";

        const cleanup = () => {
            document.body.removeChild(dialog);
            document.body.removeChild(overlay);
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
            font-size: 12px;
        `;
        cancelBtn.addEventListener("click", cleanup);

        const renameBtn = document.createElement("button");
        renameBtn.textContent = "Rename";
        renameBtn.style.cssText = `
            padding: 8px 16px;
            background: #5a5;
            border: none;
            border-radius: 4px;
            color: white;
            cursor: pointer;
            font-size: 12px;
        `;
        renameBtn.addEventListener("click", () => {
            const newName = input.value.trim();
            if (!newName || newName === info.name) {
                cleanup();
                return;
            }

            const success = variableManager.renameVariable(info.name, newName);
            if (!success) {
                errorMsg.textContent = `Variable "${newName}" already exists!`;
                return;
            }

            cleanup();
            this._lastHash = "";
            this.refresh();
        });

        input.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                renameBtn.click();
            } else if (e.key === "Escape") {
                cleanup();
            }
            e.stopPropagation();
        });

        overlay.addEventListener("click", cleanup);

        buttons.appendChild(cancelBtn);
        buttons.appendChild(renameBtn);

        dialog.appendChild(title);
        dialog.appendChild(label);
        dialog.appendChild(input);
        dialog.appendChild(errorMsg);
        dialog.appendChild(buttons);

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        input.focus();
        input.select();
    }

    // ===== Search Filter =====

    _applyFilter(text) {
        const filter = text.toLowerCase().trim();
        const rows = this.container.querySelectorAll(".nv-var-row");

        for (const row of rows) {
            const varName = row.dataset.varName || "";
            if (!filter || varName.includes(filter)) {
                row.style.display = "";
            } else {
                row.style.display = "none";
            }
        }
    }

    // ===== Navigation & Actions =====

    _navigateToNode(node) {
        if (!node || !app.canvas) return;
        app.canvas.centerOnNode(node);
        app.canvas.selectNode(node, false);
        app.canvas.setDirty(true, true);
    }

    _selectAllForVariable(info) {
        if (!app.canvas) return;

        app.canvas.deselectAll();

        // Select all getters
        for (const getter of info.getters) {
            app.canvas.selectNode(getter, true);
        }

        app.canvas.setDirty(true, true);
    }

    // ===== Drag-and-Drop: Variable from Panel to Canvas =====

    _setupVarDragListeners() {
        // Global mousemove and mouseup for drag tracking
        document.addEventListener("mousemove", (e) => {
            if (!this._varDrag.active) return;
            this._updateVarDragGhost(e);
        });

        document.addEventListener("mouseup", (e) => {
            if (!this._varDrag.active) return;
            this._endVarDrag(e);
        });
    }

    _startVarDrag(varName, varType, event) {
        this._varDrag.active = true;
        this._varDrag.varName = varName;

        // Create ghost element
        const ghost = document.createElement("div");
        const typeColor = variableManager.getTypeColor(varType);
        ghost.style.cssText = `
            position: fixed;
            left: ${event.clientX + 12}px;
            top: ${event.clientY - 10}px;
            padding: 4px 10px;
            background: #2a2a2a;
            border: 1px solid ${typeColor};
            border-radius: 4px;
            color: #e0e0e0;
            font-size: 11px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            box-shadow: 0 2px 8px rgba(0,0,0,0.5);
            z-index: 10003;
            pointer-events: none;
            white-space: nowrap;
        `;
        ghost.innerHTML = `<span style="color:${typeColor}; margin-right: 4px;">\u25CF</span> ${varName}`;
        document.body.appendChild(ghost);
        this._varDrag.ghost = ghost;
    }

    _updateVarDragGhost(event) {
        if (this._varDrag.ghost) {
            this._varDrag.ghost.style.left = `${event.clientX + 12}px`;
            this._varDrag.ghost.style.top = `${event.clientY - 10}px`;
        }
    }

    _endVarDrag(event) {
        const varName = this._varDrag.varName;

        // Remove ghost
        if (this._varDrag.ghost) {
            this._varDrag.ghost.remove();
            this._varDrag.ghost = null;
        }

        this._varDrag.active = false;
        this._varDrag.varName = null;

        if (!varName || !app.canvas) return;

        // Check if we dropped over the panel itself (ignore)
        if (this.container.contains(event.target)) return;

        // Convert screen coordinates to canvas (graph) coordinates
        let canvasPos;
        try {
            canvasPos = app.canvas.convertEventToCanvasOffset(event);
        } catch (e) {
            // Fallback: manual conversion using canvas transform
            const rect = app.canvas.canvas.getBoundingClientRect();
            const scale = app.canvas.ds.scale;
            const offset = app.canvas.ds.offset;
            canvasPos = [
                (event.clientX - rect.left) / scale - offset[0],
                (event.clientY - rect.top) / scale - offset[1],
            ];
        }

        if (!canvasPos) return;

        // Check if dropped on a node
        const targetNode = app.graph.getNodeOnPos(canvasPos[0], canvasPos[1]);

        if (targetNode) {
            // Check if dropped on an input slot
            const slotIdx = variableManager.findInputSlotAtPos(targetNode, canvasPos[0], canvasPos[1]);
            if (slotIdx >= 0) {
                variableManager.createGetterAndConnect(varName, targetNode, slotIdx);
                this._lastHash = "";
                this.refresh();
                return;
            }
        }

        // Drop on empty canvas — create getter at position
        variableManager.createGetter(varName, canvasPos);
        this._lastHash = "";
        this.refresh();
    }

    // ===== Panel Chrome =====

    _setupPanelDrag() {
        this.header.addEventListener("mousedown", (e) => {
            if (e.target.tagName === "BUTTON") return;
            this.isDragging = true;
            this.dragOffset = {
                x: e.clientX - this.position.x,
                y: e.clientY - this.position.y,
            };
            e.preventDefault();
        });

        document.addEventListener("mousemove", (e) => {
            if (!this.isDragging) return;

            this.position.x = Math.max(0, Math.min(window.innerWidth - 320, e.clientX - this.dragOffset.x));
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
        this.collapseBtn.textContent = this.isCollapsed ? "\u25BC" : "\u25B2";
        this.saveState();
    }

    show() {
        this.isVisible = true;
        this.container.style.display = 'block';
        this.saveState();
        this._lastHash = "";
        this.refresh();
    }

    hide() {
        this.isVisible = false;
        this.container.style.display = 'none';
        this.saveState();
    }

    toggle() {
        if (this.isVisible) {
            this.hide();
        } else {
            this.show();
        }
    }

    startRefreshLoop() {
        this.refreshInterval = setInterval(() => {
            if (this.isVisible && !this.isCollapsed && app.graph) {
                const newHash = variableManager.computeQuickHash();
                if (newHash !== this._lastHash) {
                    this._lastHash = newHash;
                    this.refresh();
                }
            }
        }, 1000);
    }

    destroy() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        if (this.container && this.container.parentElement) {
            this.container.parentElement.removeChild(this.container);
        }
    }
}

// Global instance
let variablesPanelInstance = null;

// Public API
window.NVVariablesPanel = {
    show: () => variablesPanelInstance?.show(),
    hide: () => variablesPanelInstance?.hide(),
    toggle: () => variablesPanelInstance?.toggle(),
    getInstance: () => variablesPanelInstance,
};

// Register extension
app.registerExtension({
    name: "NV_Comfy_Utils.VariablesPanel",

    async setup() {
        console.log("[VariablesPanel] Setting up...");

        // Wait for app to be ready
        if (!app.graph) {
            await new Promise(resolve => {
                const check = setInterval(() => {
                    if (app.graph) {
                        clearInterval(check);
                        resolve();
                    }
                }, 100);
            });
        }

        // Create the panel
        variablesPanelInstance = new VariablesPanel();

        // Add sidebar button
        try {
            const { ComfyButton } = await import("../../scripts/ui/components/button.js");

            const toggleBtn = new ComfyButton({
                icon: "swap-horizontal",
                action: () => variablesPanelInstance.toggle(),
                tooltip: "Variables",
                content: "Vars",
                classList: "comfyui-button comfyui-menu-mobile-collapse",
            });

            app.menu?.settingsGroup.element.before(toggleBtn.element);
        } catch (e) {
            console.warn("[VariablesPanel] Could not add sidebar button:", e);
        }

        // Keyboard shortcut (Ctrl+Shift+V)
        document.addEventListener("keydown", (e) => {
            if (e.ctrlKey && e.shiftKey && e.key === "V") {
                e.preventDefault();
                variablesPanelInstance.toggle();
            }
        });

        console.log("[VariablesPanel] Ready! Toggle with sidebar button or Ctrl+Shift+V");
    }
});
