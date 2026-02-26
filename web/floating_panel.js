/**
 * Floating Panel - Viewport-fixed group/node muter/bypasser
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
 */

import { app } from "../../scripts/app.js";

const MODE_ALWAYS = 0;
const MODE_BYPASS = 4;
const MODE_NEVER = 2;

const STORAGE_KEY = "NV_FloatingPanel_State";
const DEFAULT_POSITION = { x: 20, y: 100 };

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

        // Groups section (with reset order button)
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

        // Name
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
        this.saveState();
        this.refreshGroups();
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

    refreshGroups() {
        if (!app.graph || this.isDraggingGroup) return;

        const groups = app.graph._groups || [];
        this.groupsList.innerHTML = "";

        if (groups.length === 0) {
            const empty = document.createElement("div");
            empty.textContent = "No groups in workflow";
            empty.style.cssText = "padding: 8px; color: #666; font-style: italic; text-align: center;";
            this.groupsList.appendChild(empty);
            return;
        }

        // Sort groups: use custom order if available, fall back to position sort
        let sortedGroups;
        if (this.groupOrder.length > 0) {
            const orderMap = new Map(this.groupOrder.map((title, i) => [title, i]));
            sortedGroups = [...groups].sort((a, b) => {
                const aTitle = a.title || "Untitled Group";
                const bTitle = b.title || "Untitled Group";
                const aIdx = orderMap.has(aTitle) ? orderMap.get(aTitle) : Infinity;
                const bIdx = orderMap.has(bTitle) ? orderMap.get(bTitle) : Infinity;
                if (aIdx !== bIdx) return aIdx - bIdx;
                // Tie-break: position sort for groups not in custom order
                const aY = Math.floor(a._pos[1] / 30);
                const bY = Math.floor(b._pos[1] / 30);
                if (aY === bY) {
                    return Math.floor(a._pos[0] / 30) - Math.floor(b._pos[0] / 30);
                }
                return aY - bY;
            });
        } else {
            sortedGroups = [...groups].sort((a, b) => {
                const aY = Math.floor(a._pos[1] / 30);
                const bY = Math.floor(b._pos[1] / 30);
                if (aY === bY) {
                    return Math.floor(a._pos[0] / 30) - Math.floor(b._pos[0] / 30);
                }
                return aY - bY;
            });
        }

        for (const group of sortedGroups) {
            // Recompute nodes in group
            if (group.recomputeInsideNodes) {
                group.recomputeInsideNodes();
            }

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

            // Make row draggable for reordering
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
                // Remove any lingering drop indicators
                this.groupsList.querySelectorAll("[data-group-title]").forEach(el => {
                    el.style.borderTop = "";
                    el.style.borderBottom = "";
                });
            });

            row.addEventListener("dragover", (e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = "move";
                // Show drop indicator
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

                // Build current visual order from DOM
                const currentOrder = Array.from(
                    this.groupsList.querySelectorAll("[data-group-title]")
                ).map(el => el.dataset.groupTitle);

                const fromIdx = currentOrder.indexOf(draggedTitle);
                const toIdx = currentOrder.indexOf(targetTitle);
                if (fromIdx === -1 || toIdx === -1) return;

                // Determine insert position based on mouse position
                const rect = row.getBoundingClientRect();
                const insertAfter = e.clientY >= rect.top + rect.height / 2;

                // Remove dragged item and reinsert
                currentOrder.splice(fromIdx, 1);
                let insertIdx = currentOrder.indexOf(targetTitle);
                if (insertAfter) insertIdx++;
                currentOrder.splice(insertIdx, 0, draggedTitle);

                this.groupOrder = currentOrder;
                this.saveState();
                this.isDraggingGroup = false;
                this.refreshGroups();
            });

            this.groupsList.appendChild(row);
        }
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
        // Recompute which nodes are inside this group
        if (group.recomputeInsideNodes) {
            group.recomputeInsideNodes();
        }

        if (!group._nodes || group._nodes.length === 0) {
            console.warn("[FloatingPanel] No nodes in group:", group.title);
            return;
        }

        // Use BYPASS mode when disabling (not mute)
        const newMode = enable ? MODE_ALWAYS : MODE_BYPASS;
        console.log(`[FloatingPanel] Setting ${group._nodes.length} nodes in "${group.title}" to mode ${newMode}`);

        for (const node of group._nodes) {
            node.mode = newMode;
        }

        app.graph.setDirtyCanvas(true, false);
        this.refreshGroups();
    }

    navigateToGroup(group) {
        const canvas = app.canvas;
        canvas.centerOnNode(group);

        // Zoom to fit group
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
        for (const group of groups) {
            if (group.recomputeInsideNodes) {
                group.recomputeInsideNodes();
            }
            for (const node of group._nodes || []) {
                node.mode = mode;
            }
        }
        app.graph.setDirtyCanvas(true, false);
        this.refreshGroups();
    }

    // Pattern matching (similar to NodeBypasser)
    findNodesByPattern(pattern) {
        if (!app.graph) return [];

        const nodes = app.graph._nodes || [];

        // Check if it's a group pattern
        if (pattern.startsWith('@')) {
            const groupName = pattern.substring(1);
            return this.findNodesInGroup(groupName, nodes);
        }

        // Check if it's a regex pattern
        if (this.isRegexPattern(pattern)) {
            return this.findNodesByRegex(pattern, nodes);
        }

        // Simple string matching
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

    findNodesInGroup(groupName, nodes) {
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

        for (const node of nodes) {
            node.mode = newMode;
        }

        app.graph.setDirtyCanvas(true, false);
        this.refreshPatterns();
    }

    showAddPatternDialog() {
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
            z-index: 100002;
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

        const helpText = document.createElement("div");
        helpText.innerHTML = `
            <small style="color: #666;">
                Patterns: <code>*</code> = wildcard, <code>@GroupName</code> = group, <code>!</code> = exclude
            </small>
        `;
        helpText.style.marginBottom = "12px";

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
        modeSelect.innerHTML = `
            <option value="mute">Mute (completely disabled)</option>
            <option value="bypass">Bypass (pass-through)</option>
        `;

        const buttons = document.createElement("div");
        buttons.style.cssText = "display: flex; gap: 8px; justify-content: flex-end;";

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
        cancelBtn.addEventListener("click", () => {
            document.body.removeChild(dialog);
            document.body.removeChild(overlay);
        });

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

            document.body.removeChild(dialog);
            document.body.removeChild(overlay);
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
            z-index: 100001;
        `;
        overlay.addEventListener("click", () => {
            document.body.removeChild(dialog);
            document.body.removeChild(overlay);
        });

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
        // Refresh every 500ms to catch group changes
        this.refreshInterval = setInterval(() => {
            if (this.isVisible && !this.isCollapsed && app.graph) {
                this.refreshGroups();
                this.refreshPatterns();
            }
        }, 500);
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
            await new Promise((resolve, reject) => {
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
