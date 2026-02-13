/**
 * Variables Manager Panel - Dockable overlay for getter/setter variable management
 *
 * A viewport-fixed panel that provides an at-a-glance view of all
 * SetVariableNode / GetVariableNode pairs in the workflow.
 *
 * Features:
 * - Lists all variables grouped by name with setter/getter counts
 * - Type display from setter's input connection
 * - Click-to-navigate to any setter or getter node
 * - Orphan detection (getters with no matching setter)
 * - Duplicate setter warnings
 * - Search/filter bar
 * - Rename variable across all nodes
 * - Select all nodes for a variable
 * - Draggable, collapsible, state persisted to localStorage
 * - Sidebar button toggle in ComfyUI menu bar
 */

import { app } from "../../scripts/app.js";

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
        this.duplicatesSection = null;
        this.isCollapsed = false;
        this.isDragging = false;
        this.dragOffset = { x: 0, y: 0 };
        this.position = { ...DEFAULT_POSITION };
        this.refreshInterval = null;
        this.isVisible = false; // Hidden by default
        this._lastHash = "";
        this.variableMap = new Map();
        this.expandedVars = new Set();

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
                isVisible: this.isVisible
            };
            localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
        } catch (e) {
            console.warn("[VariablesPanel] Failed to save state:", e);
        }
    }

    // ===== DOM Construction =====

    createPanel() {
        // Main container - viewport fixed
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
        title.textContent = "Variables Manager";
        title.style.fontWeight = "600";

        const headerButtons = document.createElement("div");
        headerButtons.style.cssText = "display: flex; gap: 4px;";

        const refreshBtn = this.createIconButton("↻", "Refresh", () => {
            this._lastHash = "";
            this.refresh();
        });
        this.collapseBtn = this.createIconButton(this.isCollapsed ? "▼" : "▲", "Toggle collapse", () => this.toggleCollapse());
        const closeBtn = this.createIconButton("×", "Hide panel", () => this.hide());
        closeBtn.style.fontSize = "16px";

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

        // Search bar
        const searchBar = document.createElement("div");
        searchBar.style.cssText = "padding: 8px;";

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
        this.searchInput.addEventListener("input", () => this.applyFilter(this.searchInput.value));
        this.searchInput.addEventListener("focus", () => this.searchInput.style.borderColor = "#5af");
        this.searchInput.addEventListener("blur", () => this.searchInput.style.borderColor = "#444");
        searchBar.appendChild(this.searchInput);

        // Sections
        this.variablesSection = document.createElement("div");
        this.orphansSection = document.createElement("div");
        this.duplicatesSection = document.createElement("div");

        this.content.appendChild(searchBar);
        this.content.appendChild(this.variablesSection);
        this.content.appendChild(this.orphansSection);
        this.content.appendChild(this.duplicatesSection);

        this.container.appendChild(this.header);
        this.container.appendChild(this.content);

        document.body.appendChild(this.container);

        this.setupDragHandlers();
        this.refresh();
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

    createSectionHeader(title, count, warningLevel = 0) {
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

    createNavigateButton(node) {
        const btn = document.createElement("button");
        btn.textContent = "→";
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
            this.navigateToNode(node);
        });
        return btn;
    }

    // ===== Data Collection =====

    collectVariables() {
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
                    type: this.resolveType(setter),
                    isConnected: this.isSetterConnected(setter),
                    setter: setter,
                    getters: [],
                    hasOrphanGetters: false,
                    hasDuplicateSetters: false,
                    allSetters: [setter],
                });
            } else {
                const info = variableMap.get(name);
                info.hasDuplicateSetters = true;
                info.allSetters.push(setter);
            }
        }

        // Pass 2: Collect all getters, match to setters
        const getters = nodes.filter(n => n.type === "GetVariableNode");
        for (const getter of getters) {
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
                });
            }
        }

        return variableMap;
    }

    resolveType(setter) {
        if (setter.inputs && setter.inputs[0]) {
            const inputType = setter.inputs[0].type;
            return (inputType && inputType !== "*") ? inputType : "unconnected";
        }
        return "unconnected";
    }

    isSetterConnected(setter) {
        if (!setter.inputs || !setter.inputs[0]) return false;
        return setter.inputs[0].link != null;
    }

    computeQuickHash() {
        if (!app.graph) return "";
        const nodes = app.graph._nodes || [];
        let hash = "";
        for (const n of nodes) {
            if (n.type === "SetVariableNode" || n.type === "GetVariableNode") {
                hash += `${n.id}=${n.widgets?.[0]?.value}|${n.inputs?.[0]?.type}|${n.inputs?.[0]?.link},`;
            }
        }
        return hash;
    }

    // ===== Rendering =====

    refresh() {
        if (!app.graph) return;

        this.variableMap = this.collectVariables();
        this.renderVariablesSection();
        this.renderOrphansSection();
        this.renderDuplicatesSection();

        // Re-apply filter if active
        if (this.searchInput && this.searchInput.value) {
            this.applyFilter(this.searchInput.value);
        }
    }

    renderVariablesSection() {
        this.variablesSection.innerHTML = "";

        // Filter to healthy variables (have a setter, no duplicates)
        const healthy = [];
        for (const [name, info] of this.variableMap) {
            if (!info.hasOrphanGetters && !info.hasDuplicateSetters) {
                healthy.push(info);
            }
        }

        if (healthy.length === 0 && this.variableMap.size === 0) {
            const header = this.createSectionHeader("Variables", 0);
            const empty = document.createElement("div");
            empty.textContent = "No variables in workflow";
            empty.style.cssText = "padding: 12px; color: #666; font-style: italic; text-align: center;";
            this.variablesSection.appendChild(header);
            this.variablesSection.appendChild(empty);
            return;
        }

        const header = this.createSectionHeader("Variables", healthy.length);
        this.variablesSection.appendChild(header);

        for (const info of healthy) {
            const row = this.createVariableRow(info);
            this.variablesSection.appendChild(row);
        }
    }

    renderOrphansSection() {
        this.orphansSection.innerHTML = "";

        const orphans = [];
        for (const [name, info] of this.variableMap) {
            if (info.hasOrphanGetters) {
                orphans.push(info);
            }
        }

        if (orphans.length === 0) return;

        const header = this.createSectionHeader("Orphans", orphans.length, 1);
        this.orphansSection.appendChild(header);

        for (const info of orphans) {
            for (const getter of info.getters) {
                const row = this.createOrphanRow(info.name, getter);
                this.orphansSection.appendChild(row);
            }
        }
    }

    renderDuplicatesSection() {
        this.duplicatesSection.innerHTML = "";

        const duplicates = [];
        for (const [name, info] of this.variableMap) {
            if (info.hasDuplicateSetters) {
                duplicates.push(info);
            }
        }

        if (duplicates.length === 0) return;

        const header = this.createSectionHeader("Duplicates", duplicates.length, 2);
        this.duplicatesSection.appendChild(header);

        for (const info of duplicates) {
            const row = this.createDuplicateRow(info);
            this.duplicatesSection.appendChild(row);
        }
    }

    createVariableRow(info) {
        const container = document.createElement("div");
        container.classList.add("nv-var-row");
        container.dataset.varName = info.name.toLowerCase();
        container.style.cssText = `
            margin: 2px 4px;
            border-radius: 4px;
            border-left: 3px solid #3a7;
            background: #2a2a2a;
            overflow: hidden;
        `;

        const isExpanded = this.expandedVars.has(info.name);

        // Main row (clickable to expand/collapse)
        const mainRow = document.createElement("div");
        mainRow.style.cssText = `
            display: flex;
            align-items: center;
            padding: 6px 8px;
            cursor: pointer;
            transition: background 0.15s;
        `;
        mainRow.addEventListener("mouseenter", () => mainRow.style.background = "#333");
        mainRow.addEventListener("mouseleave", () => mainRow.style.background = "transparent");

        // Chevron
        const chevron = document.createElement("span");
        chevron.textContent = isExpanded ? "▾" : "▸";
        chevron.style.cssText = "margin-right: 6px; font-size: 10px; color: #888; flex-shrink: 0;";

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

        mainRow.appendChild(chevron);
        mainRow.appendChild(nameSpan);

        // Type + count info line
        const infoLine = document.createElement("div");
        infoLine.style.cssText = `
            padding: 0 8px 4px 22px;
            font-size: 10px;
            color: #888;
        `;

        const typeColor = info.isConnected ? "#5af" : "#aa5";
        const typeText = info.type;
        const getterCount = info.getters.length;
        infoLine.innerHTML = `<span style="color:${typeColor}">${typeText}</span> · 1 setter, ${getterCount} getter${getterCount !== 1 ? 's' : ''}`;

        // Expandable details
        const details = document.createElement("div");
        details.style.cssText = `
            display: ${isExpanded ? 'block' : 'none'};
            padding: 0 4px 4px 4px;
            border-top: 1px solid #333;
        `;

        // Toggle expand on main row click
        mainRow.addEventListener("click", () => {
            if (this.expandedVars.has(info.name)) {
                this.expandedVars.delete(info.name);
                details.style.display = "none";
                chevron.textContent = "▸";
            } else {
                this.expandedVars.add(info.name);
                details.style.display = "block";
                chevron.textContent = "▾";
            }
        });

        // Setter sub-row
        if (info.setter) {
            details.appendChild(this.createNodeSubRow(info.setter, "SET"));
        }

        // Getter sub-rows
        for (const getter of info.getters) {
            details.appendChild(this.createNodeSubRow(getter, "GET"));
        }

        // Action buttons
        const actions = document.createElement("div");
        actions.style.cssText = "display: flex; gap: 4px; padding: 4px 8px;";

        actions.appendChild(this.createActionButton("Select All", "#445", () => this.selectAllForVariable(info)));
        actions.appendChild(this.createActionButton("Rename", "#454", () => this.showRenameDialog(info)));

        details.appendChild(actions);

        container.appendChild(mainRow);
        container.appendChild(infoLine);
        container.appendChild(details);

        return container;
    }

    createNodeSubRow(node, role) {
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
        row.appendChild(this.createNavigateButton(node));

        return row;
    }

    createOrphanRow(varName, getter) {
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
        label.textContent = `GET "${varName}" — NO SETTER`;
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
        row.appendChild(this.createNavigateButton(getter));

        return row;
    }

    createDuplicateRow(info) {
        const container = document.createElement("div");
        container.classList.add("nv-var-row");
        container.dataset.varName = info.name.toLowerCase();
        container.style.cssText = `
            margin: 2px 4px;
            border-radius: 4px;
            border-left: 3px solid #a83;
            background: rgba(170, 136, 51, 0.1);
            overflow: hidden;
        `;

        const headerRow = document.createElement("div");
        headerRow.style.cssText = `
            display: flex;
            align-items: center;
            padding: 6px 8px;
        `;

        const icon = document.createElement("span");
        icon.textContent = "!!";
        icon.style.cssText = "color: #a83; font-weight: bold; margin-right: 8px; font-size: 11px; flex-shrink: 0;";

        const label = document.createElement("span");
        label.textContent = `"${info.name}" has ${info.allSetters.length} setters!`;
        label.style.cssText = "flex: 1; font-size: 11px; color: #ccc;";

        headerRow.appendChild(icon);
        headerRow.appendChild(label);
        container.appendChild(headerRow);

        // List each setter
        for (const setter of info.allSetters) {
            container.appendChild(this.createNodeSubRow(setter, "SET"));
        }

        // List getters too
        for (const getter of info.getters) {
            container.appendChild(this.createNodeSubRow(getter, "GET"));
        }

        return container;
    }

    // ===== Search Filter =====

    applyFilter(text) {
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

    navigateToNode(node) {
        if (!node || !app.canvas) return;
        app.canvas.centerOnNode(node);
        app.canvas.selectNode(node, false);
        app.canvas.setDirty(true, true);
    }

    selectAllForVariable(info) {
        if (!app.canvas) return;

        // Deselect all first
        app.canvas.deselectAll();

        // Select setter
        if (info.setter) {
            app.canvas.selectNode(info.setter, true); // true = add to selection
        }

        // Select all setters if duplicates
        for (const setter of info.allSetters) {
            if (setter !== info.setter) {
                app.canvas.selectNode(setter, true);
            }
        }

        // Select all getters
        for (const getter of info.getters) {
            app.canvas.selectNode(getter, true);
        }

        app.canvas.setDirty(true, true);
    }

    showRenameDialog(info) {
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
            margin-bottom: 16px;
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 4px;
            color: #e0e0e0;
            box-sizing: border-box;
            font-size: 12px;
        `;

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

            // Check for collision
            if (this.variableMap.has(newName)) {
                input.style.borderColor = "#a33";
                input.title = `Variable "${newName}" already exists!`;
                return;
            }

            this.renameVariable(info, newName);
            cleanup();
        });

        // Enter key to confirm
        input.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                renameBtn.click();
            } else if (e.key === "Escape") {
                cleanup();
            }
        });

        overlay.addEventListener("click", cleanup);

        buttons.appendChild(cancelBtn);
        buttons.appendChild(renameBtn);

        dialog.appendChild(title);
        dialog.appendChild(label);
        dialog.appendChild(input);
        dialog.appendChild(buttons);

        document.body.appendChild(overlay);
        document.body.appendChild(dialog);
        input.focus();
        input.select();
    }

    renameVariable(info, newName) {
        // Update all setters
        for (const setter of info.allSetters) {
            if (setter.widgets && setter.widgets[0]) {
                setter.widgets[0].value = newName;
            }
        }

        // Update all getters
        for (const getter of info.getters) {
            if (getter.widgets && getter.widgets[0]) {
                getter.widgets[0].value = newName;
            }
        }

        // Trigger type propagation
        if (info.setter && info.setter.updateGetters) {
            info.setter.updateGetters();
        }

        app.graph.setDirtyCanvas(true, false);
        this._lastHash = "";
        this.refresh();
    }

    // ===== Panel Chrome =====

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
        this.collapseBtn.textContent = this.isCollapsed ? "▼" : "▲";
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
                const newHash = this.computeQuickHash();
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
    getInstance: () => variablesPanelInstance
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

        // Add sidebar button to ComfyUI menu bar
        try {
            const { ComfyButton } = await import("../../scripts/ui/components/button.js");

            const toggleBtn = new ComfyButton({
                icon: "swap-horizontal",
                action: () => variablesPanelInstance.toggle(),
                tooltip: "Variables Manager",
                content: "Vars",
                classList: "comfyui-button comfyui-menu-mobile-collapse"
            });

            app.menu?.settingsGroup.element.before(toggleBtn.element);
            console.log("[VariablesPanel] Sidebar button added");
        } catch (e) {
            console.warn("[VariablesPanel] Could not add sidebar button, using keyboard shortcut only:", e);
        }

        // Keyboard shortcut fallback (Ctrl+Shift+V)
        document.addEventListener("keydown", (e) => {
            if (e.ctrlKey && e.shiftKey && e.key === "V") {
                e.preventDefault();
                variablesPanelInstance.toggle();
            }
        });

        console.log("[VariablesPanel] Ready! Toggle with sidebar button or Ctrl+Shift+V");
    }
});
