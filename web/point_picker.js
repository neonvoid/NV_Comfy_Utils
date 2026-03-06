/**
 * NV Point Picker Frontend Extension
 * Interactive canvas-based point placement for CoTracker stabilization.
 * Left-click to add a tracking point, right-click to remove nearest.
 */

import { app } from "../../scripts/app.js";

console.log("[NV_PointPicker] Loading extension...");

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
                points: [],       // [{x, y}, ...]
                hoveredIndex: -1,
                pointInfo,
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

            // Mouse events
            canvas.addEventListener("mousedown", (e) => node.handleMouseDown(e));
            canvas.addEventListener("mousemove", (e) => node.handleMouseMove(e));
            canvas.addEventListener("contextmenu", (e) => e.preventDefault());

            // Receive preview image from backend
            node.onExecuted = (message) => {
                if (message.bg_image && message.bg_image[0]) {
                    const img = new Image();
                    img.onload = () => {
                        node._pointPicker.image = img;
                        // Use actual image dimensions for coordinate accuracy
                        if (message.image_size && message.image_size[0]) {
                            node._pointPicker.imageWidth = message.image_size[0].width;
                            node._pointPicker.imageHeight = message.image_size[0].height;
                        } else {
                            node._pointPicker.imageWidth = img.naturalWidth;
                            node._pointPicker.imageHeight = img.naturalHeight;
                        }
                        canvas.width = node._pointPicker.imageWidth;
                        canvas.height = node._pointPicker.imageHeight;
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

        nodeType.prototype.handleMouseDown = function (e) {
            const { x, y } = this.mouseToCanvas(e);

            if (e.button === 2 || e.shiftKey) {
                // Right-click or shift: remove nearest point
                const nearIdx = this.findNearestPoint(x, y);
                if (nearIdx >= 0) {
                    const removed = this._pointPicker.points.splice(nearIdx, 1)[0];
                    console.log(`[NV_PointPicker] Removed point ${nearIdx + 1} at (${removed.x.toFixed(1)}, ${removed.y.toFixed(1)})`);
                    this.syncPointData();
                    this.redrawCanvas();
                }
            } else {
                // Left-click: add point
                this._pointPicker.points.push({ x, y });
                console.log(`[NV_PointPicker] Added point ${this._pointPicker.points.length} at (${x.toFixed(1)}, ${y.toFixed(1)})`);
                this.syncPointData();
                this.redrawCanvas();
            }
        };

        nodeType.prototype.handleMouseMove = function (e) {
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

            for (let i = 0; i < pp.points.length; i++) {
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
                pp.pointInfo.textContent = `${n} tracking point${n !== 1 ? "s" : ""} placed`;
            } else {
                pp.pointInfo.textContent = "Left-click: add point | Right-click: remove nearest";
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

            // Draw points
            for (let i = 0; i < points.length; i++) {
                const p = points[i];
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

                // Number label
                ctx.fillStyle = POINT_OUTLINE;
                ctx.font = `bold ${fontSize}px monospace`;
                ctx.textAlign = "left";
                ctx.fillText(`${i + 1}`, p.x + r + 4 * displayScale, p.y - r);
                ctx.fillStyle = "#fff";
                ctx.fillText(`${i + 1}`, p.x + r + 3 * displayScale, p.y - r - 1 * displayScale);
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
