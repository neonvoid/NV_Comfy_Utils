/**
 * NV BBox Creator Frontend Extension
 * Interactive canvas-based bounding box drawing with aspect ratio constraints.
 *
 * Patterns used:
 * - hideWidgetForGood() from SAM3 BBox Widget
 * - DOM widget canvas from SAM3 BBox Widget
 * - Extension registration from Frame Annotator
 */

import { app } from "../../scripts/app.js";

console.log("[NV_BBoxCreator] Loading extension...");

/**
 * Hide a widget visually while keeping serialization enabled.
 * This allows the widget to send data to backend without showing in UI.
 * IMPORTANT: Must preserve serializeValue so ComfyUI sends the value to backend.
 */
function hideWidgetForGood(node, widget) {
    if (!widget) return;

    // Save original properties
    widget.origType = widget.type;
    widget.origComputeSize = widget.computeSize;
    widget.origSerializeValue = widget.serializeValue;

    // Hide visually but keep serialization
    widget.computeSize = () => [0, -4];  // -4 compensates for litegraph's widget gap
    widget.type = "converted-widget";
    widget.hidden = true;

    // CRITICAL: Ensure serializeValue returns the actual value
    // This is what ComfyUI uses to send widget values to the backend
    widget.serializeValue = () => widget.value;

    // Hide DOM element if present
    if (widget.element) {
        widget.element.style.display = "none";
        widget.element.style.visibility = "hidden";
    }
}

app.registerExtension({
    name: "NV_Comfy_Utils.BBoxCreator",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "NV_BBoxCreator") {
            return;
        }

        console.log("[NV_BBoxCreator] Registering extension for NV_BBoxCreator");

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function() {
            // Call original if exists
            const result = onNodeCreated?.apply(this, arguments);

            const node = this;

            // Create canvas container
            const container = document.createElement("div");
            container.style.cssText = `
                position: relative;
                width: 100%;
                background: #222;
                overflow: hidden;
                box-sizing: border-box;
                display: flex;
                align-items: center;
                justify-content: center;
            `;

            // Info bar with bbox dimensions and clear button
            const infoBar = document.createElement("div");
            infoBar.style.cssText = `
                position: absolute;
                top: 5px;
                left: 5px;
                right: 5px;
                z-index: 10;
                display: flex;
                justify-content: space-between;
                align-items: center;
                pointer-events: none;
            `;
            container.appendChild(infoBar);

            // Bbox info display
            const bboxInfo = document.createElement("div");
            bboxInfo.style.cssText = `
                padding: 5px 10px;
                background: rgba(0,0,0,0.7);
                color: #fff;
                border-radius: 3px;
                font-size: 12px;
                font-family: monospace;
            `;
            bboxInfo.textContent = "Draw a bounding box";
            infoBar.appendChild(bboxInfo);

            // Clear button
            const clearBtn = document.createElement("button");
            clearBtn.textContent = "Clear";
            clearBtn.style.cssText = `
                padding: 5px 10px;
                background: #d44;
                color: #fff;
                border: 1px solid #a22;
                border-radius: 3px;
                cursor: pointer;
                font-size: 12px;
                font-weight: bold;
                pointer-events: auto;
            `;
            clearBtn.onmouseover = () => clearBtn.style.background = "#e55";
            clearBtn.onmouseout = () => clearBtn.style.background = "#d44";
            infoBar.appendChild(clearBtn);

            // Canvas element
            const canvas = document.createElement("canvas");
            canvas.width = 512;
            canvas.height = 512;
            canvas.style.cssText = `
                display: block;
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
                cursor: crosshair;
                margin: 0 auto;
            `;
            container.appendChild(canvas);

            const ctx = canvas.getContext("2d");

            // Initialize state
            node._bboxWidget = {
                canvas: canvas,
                ctx: ctx,
                container: container,
                image: null,
                bbox: null,           // Finalized bbox {x1, y1, x2, y2}
                drawing: false,
                startPoint: null,
                currentPoint: null,
                bboxInfo: bboxInfo
            };

            // Add as DOM widget
            const widget = node.addDOMWidget("canvas", "bboxCanvas", container);

            // Dynamic sizing based on node height
            widget.computeSize = (width) => {
                const nodeHeight = node.size ? node.size[1] : 480;
                const widgetHeight = Math.max(200, nodeHeight - 100);
                return [width, widgetHeight];
            };

            // Store widget reference
            node._bboxWidget.domWidget = widget;

            // Hide the bbox_data widget (delay to ensure widget is created)
            setTimeout(() => {
                const bboxDataWidget = node.widgets?.find(w => w.name === "bbox_data");
                if (bboxDataWidget) {
                    node._hiddenBboxWidget = bboxDataWidget;
                    // Ensure default value is set
                    if (!bboxDataWidget.value) {
                        bboxDataWidget.value = "{}";
                    }
                    hideWidgetForGood(node, bboxDataWidget);
                    console.log("[NV_BBoxCreator] Hidden bbox_data widget, value:", bboxDataWidget.value);
                } else {
                    console.warn("[NV_BBoxCreator] Could not find bbox_data widget. Available widgets:",
                        node.widgets?.map(w => w.name));
                }
            }, 100);

            // Clear button handler
            clearBtn.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                node._bboxWidget.bbox = null;
                node.updateBBoxData();
                node.redrawCanvas();
                console.log("[NV_BBoxCreator] Cleared bbox");
            });

            // Mouse event handlers
            canvas.addEventListener("mousedown", (e) => node.handleMouseDown(e));
            canvas.addEventListener("mousemove", (e) => node.handleMouseMove(e));
            canvas.addEventListener("mouseup", (e) => node.handleMouseUp(e));
            canvas.addEventListener("mouseleave", (e) => node.handleMouseUp(e));

            // Prevent context menu on canvas
            canvas.addEventListener("contextmenu", (e) => e.preventDefault());

            // Receive image from backend via onExecuted
            node.onExecuted = (message) => {
                console.log("[NV_BBoxCreator] onExecuted received");
                if (message.bg_image && message.bg_image[0]) {
                    const img = new Image();
                    img.onload = () => {
                        console.log(`[NV_BBoxCreator] Image loaded: ${img.width}x${img.height}`);
                        node._bboxWidget.image = img;
                        canvas.width = img.width;
                        canvas.height = img.height;
                        node.redrawCanvas();
                    };
                    img.src = "data:image/jpeg;base64," + message.bg_image[0];
                }
            };

            // Handle node resize
            const originalOnResize = node.onResize;
            node.onResize = function(size) {
                if (originalOnResize) {
                    originalOnResize.apply(this, arguments);
                }
                const containerHeight = Math.max(200, size[1] - 100);
                container.style.height = containerHeight + "px";
            };

            // Set initial size
            node.setSize([400, 500]);
            container.style.height = "400px";

            // Draw initial placeholder
            node.redrawCanvas();

            console.log("[NV_BBoxCreator] Node initialized");
            return result;
        };

        /**
         * Get current aspect ratio from widget selection.
         * Returns null for "Free" mode (no constraint).
         */
        nodeType.prototype.getAspectRatio = function() {
            const arWidget = this.widgets?.find(w => w.name === "aspect_ratio");
            const ratio = arWidget?.value || "Free";

            const presets = {
                "Free": null,
                "1:1": 1,
                "4:3": 4/3,
                "3:4": 3/4,
                "16:9": 16/9,
                "9:16": 9/16,
                "3:2": 3/2,
                "2:3": 2/3
            };

            if (ratio === "Custom") {
                const customWidget = this.widgets?.find(w => w.name === "custom_ratio");
                const customStr = customWidget?.value || "16:9";
                const parts = customStr.split(":").map(Number);
                if (parts.length === 2 && parts[0] > 0 && parts[1] > 0) {
                    return parts[0] / parts[1];
                }
                return 16/9;  // Default fallback
            }

            return presets[ratio];
        };

        /**
         * Constrain a point to maintain aspect ratio from start point.
         */
        nodeType.prototype.constrainToRatio = function(startX, startY, curX, curY) {
            const ratio = this.getAspectRatio();
            if (!ratio) return { x: curX, y: curY };

            let width = Math.abs(curX - startX);
            let height = Math.abs(curY - startY);

            // Determine constraint based on which dimension would result in larger area
            if (width / height > ratio) {
                // Width is proportionally larger, constrain it
                width = height * ratio;
            } else {
                // Height is proportionally larger, constrain it
                height = width / ratio;
            }

            // Preserve direction from start point
            const dirX = curX >= startX ? 1 : -1;
            const dirY = curY >= startY ? 1 : -1;

            return {
                x: startX + width * dirX,
                y: startY + height * dirY
            };
        };

        /**
         * Handle mouse down - start drawing bbox.
         */
        nodeType.prototype.handleMouseDown = function(e) {
            const rect = this._bboxWidget.canvas.getBoundingClientRect();
            const x = ((e.clientX - rect.left) / rect.width) * this._bboxWidget.canvas.width;
            const y = ((e.clientY - rect.top) / rect.height) * this._bboxWidget.canvas.height;

            this._bboxWidget.drawing = true;
            this._bboxWidget.startPoint = { x, y };
            this._bboxWidget.currentPoint = { x, y };

            console.log(`[NV_BBoxCreator] Start drawing at (${x.toFixed(1)}, ${y.toFixed(1)})`);
        };

        /**
         * Handle mouse move - update bbox preview with aspect ratio constraint.
         */
        nodeType.prototype.handleMouseMove = function(e) {
            if (!this._bboxWidget.drawing) return;

            const rect = this._bboxWidget.canvas.getBoundingClientRect();
            let x = ((e.clientX - rect.left) / rect.width) * this._bboxWidget.canvas.width;
            let y = ((e.clientY - rect.top) / rect.height) * this._bboxWidget.canvas.height;

            // Apply aspect ratio constraint
            const constrained = this.constrainToRatio(
                this._bboxWidget.startPoint.x,
                this._bboxWidget.startPoint.y,
                x, y
            );

            this._bboxWidget.currentPoint = constrained;
            this.redrawCanvas();
        };

        /**
         * Handle mouse up - finalize bbox.
         */
        nodeType.prototype.handleMouseUp = function(e) {
            if (!this._bboxWidget.drawing) return;

            const start = this._bboxWidget.startPoint;
            const end = this._bboxWidget.currentPoint;

            // Only save bbox if it has sufficient size
            const width = Math.abs(end.x - start.x);
            const height = Math.abs(end.y - start.y);

            if (width > 5 && height > 5) {
                this._bboxWidget.bbox = {
                    x1: Math.round(Math.min(start.x, end.x)),
                    y1: Math.round(Math.min(start.y, end.y)),
                    x2: Math.round(Math.max(start.x, end.x)),
                    y2: Math.round(Math.max(start.y, end.y))
                };
                this.updateBBoxData();
                console.log(`[NV_BBoxCreator] BBox finalized:`, this._bboxWidget.bbox);
            }

            this._bboxWidget.drawing = false;
            this._bboxWidget.startPoint = null;
            this._bboxWidget.currentPoint = null;
            this.redrawCanvas();
        };

        /**
         * Sync bbox data to hidden widget for backend.
         */
        nodeType.prototype.updateBBoxData = function() {
            const bboxJson = JSON.stringify(this._bboxWidget.bbox || {});

            if (this._hiddenBboxWidget) {
                this._hiddenBboxWidget.value = bboxJson;
                console.log("[NV_BBoxCreator] Updated bbox_data widget:", bboxJson);
            } else {
                // Try to find widget again (might have been created after initial search)
                const widget = this.widgets?.find(w => w.name === "bbox_data");
                if (widget) {
                    this._hiddenBboxWidget = widget;
                    widget.value = bboxJson;
                    console.log("[NV_BBoxCreator] Found and updated bbox_data widget:", bboxJson);
                } else {
                    console.warn("[NV_BBoxCreator] bbox_data widget not found!");
                }
            }

            // Update info display
            const bbox = this._bboxWidget.bbox;
            if (bbox) {
                const w = bbox.x2 - bbox.x1;
                const h = bbox.y2 - bbox.y1;
                const ratio = this.getAspectRatio();
                const ratioStr = ratio ? ` (${(w/h).toFixed(2)})` : "";
                this._bboxWidget.bboxInfo.textContent = `${w}x${h} at (${bbox.x1}, ${bbox.y1})${ratioStr}`;
            } else {
                this._bboxWidget.bboxInfo.textContent = "Draw a bounding box";
            }
        };

        /**
         * Redraw the canvas with image and bbox overlays.
         */
        nodeType.prototype.redrawCanvas = function() {
            const { canvas, ctx, image, bbox, drawing, startPoint, currentPoint } = this._bboxWidget;

            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw image or placeholder
            if (image) {
                ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
            } else {
                ctx.fillStyle = "#333";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = "#888";
                ctx.font = "16px sans-serif";
                ctx.textAlign = "center";
                ctx.fillText("Run node to load image", canvas.width / 2, canvas.height / 2);
                ctx.fillText("Then draw a bounding box", canvas.width / 2, canvas.height / 2 + 25);
            }

            // Draw finalized bbox (green solid)
            if (bbox) {
                const w = bbox.x2 - bbox.x1;
                const h = bbox.y2 - bbox.y1;

                // Semi-transparent fill
                ctx.fillStyle = "rgba(0, 255, 0, 0.15)";
                ctx.fillRect(bbox.x1, bbox.y1, w, h);

                // Solid border
                ctx.strokeStyle = "#0f0";
                ctx.lineWidth = 2;
                ctx.strokeRect(bbox.x1, bbox.y1, w, h);

                // Corner markers
                const markerSize = 8;
                ctx.fillStyle = "#0f0";
                // Top-left
                ctx.fillRect(bbox.x1 - markerSize/2, bbox.y1 - markerSize/2, markerSize, markerSize);
                // Top-right
                ctx.fillRect(bbox.x2 - markerSize/2, bbox.y1 - markerSize/2, markerSize, markerSize);
                // Bottom-left
                ctx.fillRect(bbox.x1 - markerSize/2, bbox.y2 - markerSize/2, markerSize, markerSize);
                // Bottom-right
                ctx.fillRect(bbox.x2 - markerSize/2, bbox.y2 - markerSize/2, markerSize, markerSize);
            }

            // Draw in-progress bbox (yellow dotted)
            if (drawing && startPoint && currentPoint) {
                const w = currentPoint.x - startPoint.x;
                const h = currentPoint.y - startPoint.y;

                // Semi-transparent fill
                ctx.fillStyle = "rgba(255, 255, 0, 0.1)";
                ctx.fillRect(startPoint.x, startPoint.y, w, h);

                // Dotted border
                ctx.strokeStyle = "#ff0";
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                ctx.strokeRect(startPoint.x, startPoint.y, w, h);
                ctx.setLineDash([]);

                // Show dimensions while drawing
                const absW = Math.abs(w);
                const absH = Math.abs(h);
                if (absW > 30 && absH > 20) {
                    ctx.fillStyle = "rgba(0,0,0,0.7)";
                    ctx.fillRect(startPoint.x + w/2 - 30, startPoint.y + h/2 - 10, 60, 20);
                    ctx.fillStyle = "#ff0";
                    ctx.font = "12px monospace";
                    ctx.textAlign = "center";
                    ctx.fillText(`${Math.round(absW)}x${Math.round(absH)}`, startPoint.x + w/2, startPoint.y + h/2 + 4);
                }
            }
        };

        // Override serialize/configure for state persistence
        nodeType.prototype.onSerialize = function(data) {
            if (this._bboxWidget?.bbox) {
                data.bboxCreatorState = {
                    bbox: this._bboxWidget.bbox
                };
            }
        };

        const originalConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function(data) {
            if (originalConfigure) {
                originalConfigure.apply(this, arguments);
            }

            if (data.bboxCreatorState?.bbox) {
                // Restore bbox after widgets are initialized
                setTimeout(() => {
                    if (this._bboxWidget) {
                        this._bboxWidget.bbox = data.bboxCreatorState.bbox;
                        this.updateBBoxData();
                        this.redrawCanvas();
                    }
                }, 100);
            }
        };
    }
});

console.log("[NV_BBoxCreator] Extension loaded");
