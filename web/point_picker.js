/**
 * NV Point Picker Frontend Extension
 * Interactive canvas-based point placement for CoTracker stabilization.
 * Left-click to add a tracking point, right-click to remove nearest.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("[NV_PointPicker] Loading extension...");

// Global execution-error listener: if the backend rejects the prompt or the queue
// fails before onExecuted fires, our local `queueInFlight` flag would stick true
// forever and lock the nav buttons. Catch the error event and reset all picker
// instances. Attach exactly once. Multi-AI review HIGH #4.
if (!window._NV_PointPicker_errorListenerAttached) {
    const resetAllInFlight = (reason) => {
        try {
            for (const n of (app.graph?._nodes || [])) {
                if (n?._pointPicker) {
                    n._pointPicker.queueInFlight = false;
                    if (typeof n._pointPicker.updateNavEnabledState === "function") {
                        n._pointPicker.updateNavEnabledState();
                    }
                }
            }
        } catch (e) {
            console.warn("[NV_PointPicker] error-listener reset failed:", e);
        }
    };
    api.addEventListener("execution_error", (e) => resetAllInFlight("execution_error"));
    api.addEventListener("execution_interrupted", (e) => resetAllInFlight("execution_interrupted"));
    window._NV_PointPicker_errorListenerAttached = true;
}

function hideWidgetForGood(node, widget) {
    if (!widget) return;
    widget.origType = widget.type;
    widget.origComputeSize = widget.computeSize;
    widget.computeSize = () => [0, -4];
    widget.type = "converted-widget";
    widget.hidden = true;
    widget.serializeValue = () => widget.value;
    if (widget.element) {
        widget.element.style.display = "none";
        widget.element.style.visibility = "hidden";
    }
}

const POINT_RADIUS = 6;
const POINT_COLOR = "#00ff88";
const POINT_OUTLINE = "#000";
const HOVER_COLOR = "#ffcc00";
const REMOVE_RADIUS = 20; // px distance to snap-remove

app.registerExtension({
    name: "NV_Comfy_Utils.PointPicker",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "NV_PointPicker") return;

        console.log("[NV_PointPicker] Registering extension");
        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;

            // Container
            const container = document.createElement("div");
            container.style.cssText = `
                position: relative; width: 100%; background: #222;
                overflow: hidden; box-sizing: border-box;
                display: flex; flex-direction: column;
            `;

            // Info bar
            const infoBar = document.createElement("div");
            infoBar.style.cssText = `
                position: relative; padding: 5px 10px; z-index: 10;
                display: flex; justify-content: space-between; align-items: center;
                background: rgba(0,0,0,0.5); flex-shrink: 0;
            `;
            container.appendChild(infoBar);

            // Point count display
            const pointInfo = document.createElement("div");
            pointInfo.style.cssText = `
                padding: 5px 10px; background: rgba(0,0,0,0.7); color: #fff;
                border-radius: 3px; font-size: 12px; font-family: monospace;
            `;
            pointInfo.textContent = "Left-click: add point | Right-click: remove nearest";
            infoBar.appendChild(pointInfo);

            // Filter toggle: show only points anchored to the current frame.
            // With multi-anchor placements (10+ points across many keyframes), the canvas
            // gets so cluttered with overlapping dots and labels that you can't see the
            // image. Toggle ON = only show points where p.t === currentFrame; existing
            // points on other frames are hidden (not deleted).
            const filterBtn = document.createElement("button");
            filterBtn.textContent = "👁 Show: All";
            filterBtn.title = "Toggle: show only points anchored to the current frame, or all points across all frames";
            filterBtn.style.cssText = `
                padding: 5px 10px; background: #444; color: #fff;
                border: 1px solid #666; border-radius: 3px; cursor: pointer;
                font-size: 12px; font-family: monospace;
            `;
            filterBtn.onmouseover = () => filterBtn.style.background = "#555";
            filterBtn.onmouseout = () => {
                // Background depends on toggle state — restored in updateFilterBtnUI
                filterBtn.style.background = node._pointPicker?.filterByCurrentFrame ? "#2a6" : "#444";
            };
            infoBar.appendChild(filterBtn);

            // Clear button
            const clearBtn = document.createElement("button");
            clearBtn.textContent = "Clear All";
            clearBtn.style.cssText = `
                padding: 5px 10px; background: #d44; color: #fff;
                border: 1px solid #a22; border-radius: 3px; cursor: pointer;
                font-size: 12px; font-weight: bold;
            `;
            clearBtn.onmouseover = () => clearBtn.style.background = "#e55";
            clearBtn.onmouseout = () => clearBtn.style.background = "#d44";
            infoBar.appendChild(clearBtn);

            // ── Frame navigation row (Prev / Next / Frame label / Goto) ──
            // Lets the user advance frames without right-click → Queue Selected Output.
            // Each button updates the frame_index widget value and triggers a queue.
            const navBar = document.createElement("div");
            navBar.style.cssText = `
                position: relative; padding: 4px 8px; z-index: 10;
                display: flex; gap: 4px; align-items: center;
                background: rgba(0,0,0,0.55); flex-shrink: 0;
                border-top: 1px solid rgba(255,255,255,0.1);
            `;
            const navBtnStyle = `
                padding: 3px 10px; background: #444; color: #fff;
                border: 1px solid #666; border-radius: 3px; cursor: pointer;
                font-size: 12px; font-family: monospace; min-width: 28px;
            `;
            const prevBtn = document.createElement("button");
            prevBtn.textContent = "⏪";
            prevBtn.title = "Previous frame (queues to refresh preview)";
            prevBtn.style.cssText = navBtnStyle;
            const nextBtn = document.createElement("button");
            nextBtn.textContent = "⏩";
            nextBtn.title = "Next frame (queues to refresh preview)";
            nextBtn.style.cssText = navBtnStyle;
            const frameLabel = document.createElement("span");
            frameLabel.style.cssText = `
                font-size: 11px; font-family: monospace; color: #aaa;
                min-width: 100px; text-align: center;
            `;
            frameLabel.textContent = "Frame: ?";
            const gotoInput = document.createElement("input");
            gotoInput.type = "number";
            gotoInput.min = 0;
            gotoInput.placeholder = "Go to";
            gotoInput.title = "Type a frame number and press Enter to jump (queues to refresh)";
            gotoInput.style.cssText = `
                width: 60px; padding: 3px 6px; background: #222; color: #fff;
                border: 1px solid #555; border-radius: 3px; font-size: 11px;
                font-family: monospace;
            `;
            navBar.appendChild(prevBtn);
            navBar.appendChild(nextBtn);
            navBar.appendChild(frameLabel);
            const navSpacer = document.createElement("div");
            navSpacer.style.flex = "1";
            navBar.appendChild(navSpacer);
            navBar.appendChild(gotoInput);
            container.appendChild(navBar);

            // Canvas wrapper
            const canvasWrapper = document.createElement("div");
            canvasWrapper.style.cssText = `
                flex: 1; display: flex; align-items: center; justify-content: center;
                overflow: hidden; min-height: 200px;
            `;
            container.appendChild(canvasWrapper);

            // Canvas
            const canvas = document.createElement("canvas");
            canvas.width = 512;
            canvas.height = 512;
            canvas.style.cssText = `
                display: block; max-width: 100%; max-height: 100%;
                object-fit: contain; cursor: crosshair; margin: 0 auto;
            `;
            canvasWrapper.appendChild(canvas);
            const ctx = canvas.getContext("2d");

            // State
            node._pointPicker = {
                canvas, ctx, container, canvasWrapper,
                image: null,
                imageWidth: 512,
                imageHeight: 512,
                points: [],       // [{x, y, t}, ...]
                hoveredIndex: -1,
                pointInfo,
                // Filter state — when true, only render points where p.t === current frame
                filterByCurrentFrame: false,
                filterBtn,
                // Frame navigation state
                lastBackendFrameIndex: null,   // authoritative frame_index that backend used (set in onExecuted)
                totalFrames: 0,                // total batch size, populated by backend
                frameLabel,                    // navBar UI elements
                gotoInput, prevBtn, nextBtn,
            };

            // DOM widget
            const widget = node.addDOMWidget("canvas", "pointPickerCanvas", container);
            widget.computeSize = (width) => [width, 400];
            node._pointPicker.domWidget = widget;

            // Hide point_data widget
            setTimeout(() => {
                const pdWidget = node.widgets?.find(w => w.name === "point_data");
                if (pdWidget) {
                    node._hiddenPointWidget = pdWidget;
                    if (!pdWidget.value) pdWidget.value = "[]";
                    hideWidgetForGood(node, pdWidget);
                }
            }, 100);

            // Clear handler
            clearBtn.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                node._pointPicker.points = [];
                node.syncPointData();
                node.redrawCanvas();
            });

            // Filter toggle handler — flip filterByCurrentFrame, update button cosmetics, redraw.
            const updateFilterBtnUI = () => {
                const pp = node._pointPicker;
                if (!pp) return;
                if (pp.filterByCurrentFrame) {
                    pp.filterBtn.textContent = "👁 Show: This Frame";
                    pp.filterBtn.style.background = "#2a6";
                    pp.filterBtn.style.borderColor = "#1a4";
                } else {
                    pp.filterBtn.textContent = "👁 Show: All";
                    pp.filterBtn.style.background = "#444";
                    pp.filterBtn.style.borderColor = "#666";
                }
            };
            node._pointPicker.updateFilterBtnUI = updateFilterBtnUI;
            updateFilterBtnUI();
            filterBtn.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                node._pointPicker.filterByCurrentFrame = !node._pointPicker.filterByCurrentFrame;
                updateFilterBtnUI();
                node.syncPointData();   // refresh the count display in pointInfo
                node.redrawCanvas();
            });
            filterBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
            filterBtn.addEventListener("mousedown", (e) => e.stopPropagation());

            // ── Frame navigation handlers ──
            // Detect whether frame_index is driven by an upstream connection (wired-as-input).
            // If so, our local widget mutation won't affect the actual execution value because
            // ComfyUI evaluates the upstream node and ignores the widget. Disable nav in that case.
            const isFrameIndexWired = () => {
                // The widget exists but its `type` becomes "converted-widget" when wired.
                // Also check the input slot for a link.
                const fiWidget = node.widgets?.find(w => w.name === "frame_index");
                if (fiWidget && (fiWidget.type === "converted-widget" || fiWidget.hidden === true)) {
                    return true;
                }
                const fiInput = node.inputs?.find(i => i.name === "frame_index");
                if (fiInput && fiInput.link != null) {
                    return true;
                }
                return false;
            };

            const updateNavEnabledState = () => {
                const wired = isFrameIndexWired();
                const inFlight = !!node._pointPicker.queueInFlight;
                const disabled = wired || inFlight;
                prevBtn.disabled = disabled;
                nextBtn.disabled = disabled;
                gotoInput.disabled = disabled;
                prevBtn.style.opacity = disabled ? "0.5" : "1";
                nextBtn.style.opacity = disabled ? "0.5" : "1";
                gotoInput.style.opacity = disabled ? "0.5" : "1";
                if (wired && node._pointPicker.frameLabel) {
                    node._pointPicker.frameLabel.textContent = "Frame driven by upstream input — nav disabled";
                }
            };
            node._pointPicker.updateNavEnabledState = updateNavEnabledState;
            // Initial state — also re-check when connections change
            updateNavEnabledState();
            const origOnConnectionsChange = node.onConnectionsChange;
            node.onConnectionsChange = function (...args) {
                if (origOnConnectionsChange) origOnConnectionsChange.apply(this, args);
                updateNavEnabledState();
            };

            // seekFrame: bump frame_index widget, then queue ONLY this node's subgraph (not the
            // whole workflow). app.queueOutputNodes([id]) is the same path as right-click → "Queue
            // Selected Output Nodes" — runs this picker + its upstream image source, nothing else.
            // Falls back to a clear warning if the API isn't available rather than silently
            // running the entire (potentially expensive) workflow.
            const partialQueueThisNode = () => {
                // Preferred: ComfyUI 1.x partial-queue API
                if (typeof app.queueOutputNodes === "function") {
                    return app.queueOutputNodes([node.id]);
                }
                // Fallback: temporarily select only this node and run the menu command
                if (app.commandManager?.executeCommand && app.canvas) {
                    const prevSelected = Object.values(app.canvas.selected_nodes || {});
                    app.canvas.deselectAllNodes?.();
                    app.canvas.selectNode?.(node, false);
                    try {
                        return app.commandManager.executeCommand("Comfy.Canvas.QueueSelectedOutputNodes");
                    } finally {
                        app.canvas.deselectAllNodes?.();
                        for (const n of prevSelected) {
                            if (n !== node) app.canvas.selectNode?.(n, true);
                        }
                    }
                }
                // No partial-queue path available — refuse to run the full workflow silently.
                throw new Error(
                    "No partial-queue API available (app.queueOutputNodes / commandManager.executeCommand). " +
                    "Right-click the node → Queue Selected Output Nodes to refresh the preview manually."
                );
            };

            const seekFrame = (delta, absolute = null) => {
                if (isFrameIndexWired()) {
                    console.warn("[NV_PointPicker] frame_index is wired as input — disconnect to use nav buttons");
                    return;
                }
                const fiWidget = node.widgets?.find(w => w.name === "frame_index");
                if (!fiWidget) return;
                const total = node._pointPicker.totalFrames || 1;
                const current = (typeof node._pointPicker.lastBackendFrameIndex === "number"
                                  ? node._pointPicker.lastBackendFrameIndex
                                  : Number(fiWidget.value) || 0);
                let next;
                if (absolute !== null) {
                    next = Math.max(0, Math.min(absolute, total - 1));
                } else {
                    next = ((current + delta) % total + total) % total;
                }
                fiWidget.value = next;
                if (fiWidget.callback) fiWidget.callback(next);
                node._pointPicker.frameLabel.textContent = `Frame: ${next + 1} / ${total} (queueing...)`;
                // Mark queue-in-flight to disable buttons until onExecuted clears it
                node._pointPicker.queueInFlight = true;
                updateNavEnabledState();
                try {
                    partialQueueThisNode();
                } catch (err) {
                    console.warn("[NV_PointPicker] partial queue failed:", err);
                    node._pointPicker.frameLabel.textContent = `Frame: ${next + 1} / ${total} — right-click → Queue Selected to refresh`;
                    // Recover so buttons aren't permanently disabled
                    node._pointPicker.queueInFlight = false;
                    updateNavEnabledState();
                }
            };

            prevBtn.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                seekFrame(-1);
            });
            nextBtn.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                seekFrame(1);
            });
            gotoInput.addEventListener("keydown", (e) => {
                if (e.key === "Enter") {
                    e.preventDefault();
                    e.stopPropagation();
                    // gotoInput shows 1-indexed frame numbers (Frame X / Y label is 1-indexed),
                    // so subtract 1 to get 0-indexed value passed to seekFrame.
                    const target1 = parseInt(gotoInput.value, 10);
                    if (Number.isFinite(target1)) seekFrame(0, target1 - 1);
                }
            });
            // Stop button presses from being interpreted as canvas-area events by LiteGraph
            for (const el of [prevBtn, nextBtn, gotoInput]) {
                el.addEventListener("pointerdown", (e) => e.stopPropagation());
                el.addEventListener("mousedown", (e) => e.stopPropagation());
            }

            // Mouse events
            canvas.addEventListener("mousedown", (e) => node.handleMouseDown(e));
            canvas.addEventListener("mousemove", (e) => node.handleMouseMove(e));
            canvas.addEventListener("contextmenu", (e) => e.preventDefault());

            // Receive preview image + frame info from backend.
            // Chain any prior onExecuted handler so we don't silently drop other extensions' hooks.
            const priorOnExecuted = node.onExecuted;
            node.onExecuted = (message) => {
                if (priorOnExecuted) {
                    try { priorOnExecuted.call(node, message); }
                    catch (e) { console.warn("[NV_PointPicker] prior onExecuted threw:", e); }
                }
                const pp = node._pointPicker;
                if (!pp) return;
                // Clear queue-in-flight so nav buttons re-enable
                pp.queueInFlight = false;
                if (pp.updateNavEnabledState) pp.updateNavEnabledState();

                // Capture authoritative frame_index from backend — handles wired-input case
                // where the widget's .value goes stale (the bug that caused the t-collapse).
                if (message.frame_index && typeof message.frame_index[0] === "number") {
                    pp.lastBackendFrameIndex = message.frame_index[0];
                }
                if (message.total_frames && typeof message.total_frames[0] === "number") {
                    pp.totalFrames = message.total_frames[0];
                    if (pp.gotoInput) {
                        pp.gotoInput.max = Math.max(1, pp.totalFrames);
                    }
                }
                // Update Frame: X / Y label (1-indexed for display)
                if (pp.frameLabel && !isFrameIndexWired()) {
                    const cur = (typeof pp.lastBackendFrameIndex === "number")
                        ? (pp.lastBackendFrameIndex + 1) : "?";
                    const tot = pp.totalFrames || "?";
                    pp.frameLabel.textContent = `Frame: ${cur} / ${tot}`;
                }

                // When filter is active and frame just changed, refresh the
                // "X on this frame, Y hidden" count in the header.
                if (pp.filterByCurrentFrame && typeof node.syncPointData === "function") {
                    node.syncPointData();
                }
                if (message.bg_image && message.bg_image[0]) {
                    const img = new Image();
                    img.onload = () => {
                        pp.image = img;
                        // Use actual image dimensions for coordinate accuracy
                        if (message.image_size && message.image_size[0]) {
                            pp.imageWidth = message.image_size[0].width;
                            pp.imageHeight = message.image_size[0].height;
                        } else {
                            pp.imageWidth = img.naturalWidth;
                            pp.imageHeight = img.naturalHeight;
                        }
                        canvas.width = pp.imageWidth;
                        canvas.height = pp.imageHeight;
                        node.redrawCanvas();
                    };
                    img.src = "data:image/jpeg;base64," + message.bg_image[0];
                }
            };

            node.setSize([400, 520]);
            container.style.height = "400px";
            node.redrawCanvas();
            return result;
        };

        // =====================================================================
        // Mouse handlers
        // =====================================================================

        // Map mouse event to canvas pixel coordinates, accounting for
        // object-fit: contain letterboxing
        nodeType.prototype.mouseToCanvas = function (e) {
            const pp = this._pointPicker;
            const rect = pp.canvas.getBoundingClientRect();

            // Compute the actual rendered image area within the element
            const elemW = rect.width;
            const elemH = rect.height;
            const canvasAspect = pp.canvas.width / pp.canvas.height;
            const elemAspect = elemW / elemH;

            let renderW, renderH, offsetX, offsetY;
            if (canvasAspect > elemAspect) {
                // Canvas is wider than element — letterbox top/bottom
                renderW = elemW;
                renderH = elemW / canvasAspect;
                offsetX = 0;
                offsetY = (elemH - renderH) / 2;
            } else {
                // Canvas is taller than element — letterbox left/right
                renderH = elemH;
                renderW = elemH * canvasAspect;
                offsetX = (elemW - renderW) / 2;
                offsetY = 0;
            }

            const x = ((e.clientX - rect.left - offsetX) / renderW) * pp.canvas.width;
            const y = ((e.clientY - rect.top - offsetY) / renderH) * pp.canvas.height;
            return { x, y };
        };

        nodeType.prototype.getCurrentFrameIndex = function () {
            // Each new point gets stamped with the frame currently displayed in the picker.
            // Priority order:
            //   1. lastBackendFrameIndex — authoritative value the backend last used to
            //      render the preview. Set in onExecuted from message.frame_index.
            //      Handles the wired-input case correctly (Int upstream node feeding
            //      frame_index — widget.value goes stale, but backend echo is correct).
            //   2. frame_index widget value — fallback when backend hasn't replied yet
            //      (e.g. very first click before any execute).
            //   3. 0 — last-resort fallback.
            const pp = this._pointPicker;
            if (pp && typeof pp.lastBackendFrameIndex === "number" && Number.isFinite(pp.lastBackendFrameIndex)) {
                return Math.max(0, Math.floor(pp.lastBackendFrameIndex));
            }
            const fiWidget = this.widgets?.find(w => w.name === "frame_index");
            const v = fiWidget ? Number(fiWidget.value) : 0;
            return Number.isFinite(v) && v >= 0 ? Math.floor(v) : 0;
        };

        nodeType.prototype.handleMouseDown = function (e) {
            const { x, y } = this.mouseToCanvas(e);

            if (e.button === 2 || e.shiftKey) {
                // Right-click or shift: remove nearest point
                const nearIdx = this.findNearestPoint(x, y);
                if (nearIdx >= 0) {
                    const removed = this._pointPicker.points.splice(nearIdx, 1)[0];
                    const tStr = removed.t !== undefined ? `@t=${removed.t}` : "";
                    console.log(`[NV_PointPicker] Removed point ${nearIdx + 1} at (${removed.x.toFixed(1)}, ${removed.y.toFixed(1)})${tStr}`);
                    this.syncPointData();
                    this.redrawCanvas();
                }
            } else {
                // Left-click: add point with current frame_index as anchor t
                const t = this.getCurrentFrameIndex();
                this._pointPicker.points.push({ x, y, t });
                console.log(`[NV_PointPicker] Added point ${this._pointPicker.points.length} at (${x.toFixed(1)}, ${y.toFixed(1)})@t=${t}`);
                this.syncPointData();
                this.redrawCanvas();
            }
        };

        nodeType.prototype.handleMouseMove = function (e) {
            // Pre-existing bug: `pp` was used without being declared, causing a silent
            // ReferenceError on every mousemove (browsers swallow event-handler errors).
            // Caught by Codex multi-AI review 2026-04-30.
            const pp = this._pointPicker;
            if (!pp) return;
            const { x, y } = this.mouseToCanvas(e);

            const oldHover = pp.hoveredIndex;
            pp.hoveredIndex = this.findNearestPoint(x, y);
            if (pp.hoveredIndex !== oldHover) {
                this.redrawCanvas();
            }
        };

        nodeType.prototype.findNearestPoint = function (x, y) {
            const pp = this._pointPicker;
            let bestIdx = -1;
            let bestDist = REMOVE_RADIUS;
            // Scale threshold by actual rendered size (not element size)
            const rect = pp.canvas.getBoundingClientRect();
            const canvasAspect = pp.canvas.width / pp.canvas.height;
            const elemAspect = rect.width / rect.height;
            const renderW = canvasAspect > elemAspect ? rect.width : rect.height * canvasAspect;
            const scale = pp.canvas.width / renderW;
            const threshold = REMOVE_RADIUS * scale;

            // When filter is on, only consider points on the current frame — otherwise
            // a right-click could silently remove a point you can't even see.
            const filterFrame = pp.filterByCurrentFrame ? this.getCurrentFrameIndex() : null;

            for (let i = 0; i < pp.points.length; i++) {
                if (filterFrame !== null && pp.points[i].t !== filterFrame) continue;
                const dx = pp.points[i].x - x;
                const dy = pp.points[i].y - y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < threshold && dist < bestDist) {
                    bestDist = dist;
                    bestIdx = i;
                }
            }
            return bestIdx;
        };

        // =====================================================================
        // Sync & draw
        // =====================================================================

        nodeType.prototype.syncPointData = function () {
            const pp = this._pointPicker;
            const json = JSON.stringify(pp.points);

            if (this._hiddenPointWidget) {
                this._hiddenPointWidget.value = json;
            } else {
                const w = this.widgets?.find(w => w.name === "point_data");
                if (w) {
                    this._hiddenPointWidget = w;
                    w.value = json;
                }
            }

            const n = pp.points.length;
            if (n > 0) {
                const ts = pp.points.map(p => p.t).filter(t => t !== undefined && t !== null);
                const uniq = [...new Set(ts)].sort((a, b) => a - b);
                const anchorStr = uniq.length === 1 ? `t=${uniq[0]}`
                                : uniq.length > 1 ? `multi-anchor t=${uniq.join(",")}`
                                : "";
                let header = `${n} tracking point${n !== 1 ? "s" : ""} placed${anchorStr ? ` (${anchorStr})` : ""}`;
                // When filter is active, show "X on this frame / Y hidden"
                if (pp.filterByCurrentFrame) {
                    const cur = this.getCurrentFrameIndex?.();
                    const onFrame = pp.points.filter(p => p.t === cur).length;
                    header += ` — showing ${onFrame} on frame ${cur}, ${n - onFrame} hidden`;
                }
                pp.pointInfo.textContent = header;
            } else {
                pp.pointInfo.textContent = "Left-click: add point on current frame | Change frame_index then click for multi-anchor";
            }
        };

        nodeType.prototype.redrawCanvas = function () {
            const pp = this._pointPicker;
            const { canvas, ctx, image, points, hoveredIndex } = pp;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Background image or placeholder
            if (image) {
                ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
            } else {
                ctx.fillStyle = "#333";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = "#888";
                ctx.font = "16px sans-serif";
                ctx.textAlign = "center";
                ctx.fillText("Run node to load preview", canvas.width / 2, canvas.height / 2 - 10);
                ctx.fillText("Then click to place tracking points", canvas.width / 2, canvas.height / 2 + 14);
            }

            // Scale point radius based on image size so points are visible on large images
            const displayScale = Math.max(1, Math.min(canvas.width, canvas.height) / 400);
            const r = POINT_RADIUS * displayScale;
            const fontSize = Math.round(11 * displayScale);

            // Filter: when filterByCurrentFrame is on, only render points where p.t === current frame
            const filterFrame = pp.filterByCurrentFrame
                ? (typeof this.getCurrentFrameIndex === "function" ? this.getCurrentFrameIndex() : null)
                : null;

            // Draw points
            for (let i = 0; i < points.length; i++) {
                const p = points[i];
                if (filterFrame !== null && p.t !== filterFrame) continue;  // hidden by filter
                const isHovered = (i === hoveredIndex);
                const color = isHovered ? HOVER_COLOR : POINT_COLOR;

                // Outer ring
                ctx.beginPath();
                ctx.arc(p.x, p.y, r + 2 * displayScale, 0, Math.PI * 2);
                ctx.fillStyle = POINT_OUTLINE;
                ctx.fill();

                // Inner dot
                ctx.beginPath();
                ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();

                // Crosshair
                const ch = r + 4 * displayScale;
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.5 * displayScale;
                ctx.beginPath();
                ctx.moveTo(p.x - ch, p.y);
                ctx.lineTo(p.x - r - 1 * displayScale, p.y);
                ctx.moveTo(p.x + r + 1 * displayScale, p.y);
                ctx.lineTo(p.x + ch, p.y);
                ctx.moveTo(p.x, p.y - ch);
                ctx.lineTo(p.x, p.y - r - 1 * displayScale);
                ctx.moveTo(p.x, p.y + r + 1 * displayScale);
                ctx.lineTo(p.x, p.y + ch);
                ctx.stroke();

                // Number + anchor-frame label: "1@t=50" so the user can see
                // which frame each point was anchored on
                const tLabel = (p.t !== undefined && p.t !== null) ? `${i + 1}@t=${p.t}` : `${i + 1}`;
                ctx.fillStyle = POINT_OUTLINE;
                ctx.font = `bold ${fontSize}px monospace`;
                ctx.textAlign = "left";
                ctx.fillText(tLabel, p.x + r + 4 * displayScale, p.y - r);
                ctx.fillStyle = "#fff";
                ctx.fillText(tLabel, p.x + r + 3 * displayScale, p.y - r - 1 * displayScale);
            }
        };

        // =====================================================================
        // State persistence
        // =====================================================================

        nodeType.prototype.onSerialize = function (data) {
            if (this._pointPicker) {
                data.pointPickerState = { points: this._pointPicker.points };
            }
        };

        const originalConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (data) {
            if (originalConfigure) originalConfigure.apply(this, arguments);
            if (data.pointPickerState) {
                setTimeout(() => {
                    if (this._pointPicker) {
                        this._pointPicker.points = data.pointPickerState.points || [];
                        this.syncPointData();
                        this.redrawCanvas();
                    }
                }, 100);
            }
        };
    },
});

console.log("[NV_PointPicker] Extension loaded");
