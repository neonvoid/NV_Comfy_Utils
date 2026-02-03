/**
 * NV BBox Creator Frontend Extension
 * Interactive canvas-based bounding box drawing with aspect ratio constraints.
 * Supports both positive (left-click, green) and negative (right-click, red) boxes.
 */

import { app } from "../../scripts/app.js";

console.log("[NV_BBoxCreator] Loading extension...");

/**
 * Hide a widget visually while keeping serialization enabled.
 */
function hideWidgetForGood(node, widget) {
    if (!widget) return;

    widget.origType = widget.type;
    widget.origComputeSize = widget.computeSize;
    widget.origSerializeValue = widget.serializeValue;

    widget.computeSize = () => [0, -4];
    widget.type = "converted-widget";
    widget.hidden = true;
    widget.serializeValue = () => widget.value;

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

            // Info bar
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
            bboxInfo.textContent = "Left-click: positive | Right-click: negative";
            infoBar.appendChild(bboxInfo);

            // Clear button
            const clearBtn = document.createElement("button");
            clearBtn.textContent = "Clear All";
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

            // Initialize state with positive AND negative boxes
            node._bboxWidget = {
                canvas: canvas,
                ctx: ctx,
                container: container,
                image: null,
                positiveBBoxes: [],    // Array of {x1, y1, x2, y2}
                negativeBBoxes: [],    // Array of {x1, y1, x2, y2}
                drawing: false,
                isNegative: false,     // Is current drawing a negative box?
                startPoint: null,
                currentPoint: null,
                bboxInfo: bboxInfo
            };

            // Add as DOM widget
            const widget = node.addDOMWidget("canvas", "bboxCanvas", container);
            widget.computeSize = (width) => [width, 400];
            node._bboxWidget.domWidget = widget;

            // Hide the bbox_data widget
            setTimeout(() => {
                const bboxDataWidget = node.widgets?.find(w => w.name === "bbox_data");
                if (bboxDataWidget) {
                    node._hiddenBboxWidget = bboxDataWidget;
                    if (!bboxDataWidget.value) {
                        bboxDataWidget.value = "{}";
                    }
                    hideWidgetForGood(node, bboxDataWidget);
                }
            }, 100);

            // Clear button handler
            clearBtn.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                node._bboxWidget.positiveBBoxes = [];
                node._bboxWidget.negativeBBoxes = [];
                node.updateBBoxData();
                node.redrawCanvas();
                console.log("[NV_BBoxCreator] Cleared all bboxes");
            });

            // Mouse event handlers
            canvas.addEventListener("mousedown", (e) => node.handleMouseDown(e));
            canvas.addEventListener("mousemove", (e) => node.handleMouseMove(e));
            canvas.addEventListener("mouseup", (e) => node.handleMouseUp(e));
            canvas.addEventListener("mouseleave", (e) => node.handleMouseUp(e));

            // Prevent context menu on canvas
            canvas.addEventListener("contextmenu", (e) => e.preventDefault());

            // Receive image from backend
            node.onExecuted = (message) => {
                if (message.bg_image && message.bg_image[0]) {
                    const img = new Image();
                    img.onload = () => {
                        node._bboxWidget.image = img;
                        canvas.width = img.width;
                        canvas.height = img.height;
                        node.redrawCanvas();
                    };
                    img.src = "data:image/jpeg;base64," + message.bg_image[0];
                }
            };

            // Set initial size
            node.setSize([400, 520]);
            container.style.height = "400px";

            node.redrawCanvas();
            return result;
        };

        /**
         * Get current aspect ratio from widget selection.
         */
        nodeType.prototype.getAspectRatio = function() {
            const arWidget = this.widgets?.find(w => w.name === "aspect_ratio");
            const ratio = arWidget?.value || "Free";

            const presets = {
                "Free": null, "1:1": 1, "4:3": 4/3, "3:4": 3/4,
                "16:9": 16/9, "9:16": 9/16, "3:2": 3/2, "2:3": 2/3
            };

            if (ratio === "Custom") {
                const customWidget = this.widgets?.find(w => w.name === "custom_ratio");
                const customStr = customWidget?.value || "16:9";
                const parts = customStr.split(":").map(Number);
                if (parts.length === 2 && parts[0] > 0 && parts[1] > 0) {
                    return parts[0] / parts[1];
                }
                return 16/9;
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

            if (width / height > ratio) {
                width = height * ratio;
            } else {
                height = width / ratio;
            }

            const dirX = curX >= startX ? 1 : -1;
            const dirY = curY >= startY ? 1 : -1;

            return {
                x: startX + width * dirX,
                y: startY + height * dirY
            };
        };

        /**
         * Handle mouse down - start drawing bbox.
         * Left-click = positive, Right-click = negative
         */
        nodeType.prototype.handleMouseDown = function(e) {
            const rect = this._bboxWidget.canvas.getBoundingClientRect();
            const x = ((e.clientX - rect.left) / rect.width) * this._bboxWidget.canvas.width;
            const y = ((e.clientY - rect.top) / rect.height) * this._bboxWidget.canvas.height;

            // Right-click (button 2) or shift+click = negative box
            const isNegative = e.button === 2 || e.shiftKey;

            this._bboxWidget.drawing = true;
            this._bboxWidget.isNegative = isNegative;
            this._bboxWidget.startPoint = { x, y };
            this._bboxWidget.currentPoint = { x, y };

            console.log(`[NV_BBoxCreator] Start drawing ${isNegative ? 'NEGATIVE' : 'POSITIVE'} at (${x.toFixed(1)}, ${y.toFixed(1)})`);
        };

        /**
         * Handle mouse move - update bbox preview.
         */
        nodeType.prototype.handleMouseMove = function(e) {
            if (!this._bboxWidget.drawing) return;

            const rect = this._bboxWidget.canvas.getBoundingClientRect();
            let x = ((e.clientX - rect.left) / rect.width) * this._bboxWidget.canvas.width;
            let y = ((e.clientY - rect.top) / rect.height) * this._bboxWidget.canvas.height;

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

            const width = Math.abs(end.x - start.x);
            const height = Math.abs(end.y - start.y);

            if (width > 5 && height > 5) {
                const bbox = {
                    x1: Math.round(Math.min(start.x, end.x)),
                    y1: Math.round(Math.min(start.y, end.y)),
                    x2: Math.round(Math.max(start.x, end.x)),
                    y2: Math.round(Math.max(start.y, end.y))
                };

                // Add to appropriate array
                if (this._bboxWidget.isNegative) {
                    this._bboxWidget.negativeBBoxes.push(bbox);
                    console.log(`[NV_BBoxCreator] Added NEGATIVE bbox:`, bbox);
                } else {
                    this._bboxWidget.positiveBBoxes.push(bbox);
                    console.log(`[NV_BBoxCreator] Added POSITIVE bbox:`, bbox);
                }

                this.updateBBoxData();
            }

            this._bboxWidget.drawing = false;
            this._bboxWidget.isNegative = false;
            this._bboxWidget.startPoint = null;
            this._bboxWidget.currentPoint = null;
            this.redrawCanvas();
        };

        /**
         * Sync bbox data to hidden widget for backend.
         */
        nodeType.prototype.updateBBoxData = function() {
            const data = {
                positive: this._bboxWidget.positiveBBoxes,
                negative: this._bboxWidget.negativeBBoxes
            };
            const bboxJson = JSON.stringify(data);

            if (this._hiddenBboxWidget) {
                this._hiddenBboxWidget.value = bboxJson;
            } else {
                const widget = this.widgets?.find(w => w.name === "bbox_data");
                if (widget) {
                    this._hiddenBboxWidget = widget;
                    widget.value = bboxJson;
                }
            }

            // Update info display
            const posCount = this._bboxWidget.positiveBBoxes.length;
            const negCount = this._bboxWidget.negativeBBoxes.length;
            if (posCount > 0 || negCount > 0) {
                this._bboxWidget.bboxInfo.textContent = `Positive: ${posCount} | Negative: ${negCount}`;
            } else {
                this._bboxWidget.bboxInfo.textContent = "Left-click: positive | Right-click: negative";
            }
        };

        /**
         * Redraw the canvas with image and bbox overlays.
         */
        nodeType.prototype.redrawCanvas = function() {
            const { canvas, ctx, image, positiveBBoxes, negativeBBoxes, drawing, isNegative, startPoint, currentPoint } = this._bboxWidget;

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
                ctx.fillText("Run node to load image", canvas.width / 2, canvas.height / 2 - 12);
                ctx.fillText("Left-click: positive (green)", canvas.width / 2, canvas.height / 2 + 12);
                ctx.fillText("Right-click: negative (red)", canvas.width / 2, canvas.height / 2 + 36);
            }

            // Draw positive bboxes (green)
            for (const bbox of positiveBBoxes) {
                const w = bbox.x2 - bbox.x1;
                const h = bbox.y2 - bbox.y1;

                ctx.fillStyle = "rgba(0, 255, 0, 0.15)";
                ctx.fillRect(bbox.x1, bbox.y1, w, h);

                ctx.strokeStyle = "#0f0";
                ctx.lineWidth = 2;
                ctx.strokeRect(bbox.x1, bbox.y1, w, h);

                // Corner markers
                const markerSize = 6;
                ctx.fillStyle = "#0f0";
                ctx.fillRect(bbox.x1 - markerSize/2, bbox.y1 - markerSize/2, markerSize, markerSize);
                ctx.fillRect(bbox.x2 - markerSize/2, bbox.y1 - markerSize/2, markerSize, markerSize);
                ctx.fillRect(bbox.x1 - markerSize/2, bbox.y2 - markerSize/2, markerSize, markerSize);
                ctx.fillRect(bbox.x2 - markerSize/2, bbox.y2 - markerSize/2, markerSize, markerSize);
            }

            // Draw negative bboxes (red)
            for (const bbox of negativeBBoxes) {
                const w = bbox.x2 - bbox.x1;
                const h = bbox.y2 - bbox.y1;

                ctx.fillStyle = "rgba(255, 0, 0, 0.15)";
                ctx.fillRect(bbox.x1, bbox.y1, w, h);

                ctx.strokeStyle = "#f00";
                ctx.lineWidth = 2;
                ctx.strokeRect(bbox.x1, bbox.y1, w, h);

                // Corner markers
                const markerSize = 6;
                ctx.fillStyle = "#f00";
                ctx.fillRect(bbox.x1 - markerSize/2, bbox.y1 - markerSize/2, markerSize, markerSize);
                ctx.fillRect(bbox.x2 - markerSize/2, bbox.y1 - markerSize/2, markerSize, markerSize);
                ctx.fillRect(bbox.x1 - markerSize/2, bbox.y2 - markerSize/2, markerSize, markerSize);
                ctx.fillRect(bbox.x2 - markerSize/2, bbox.y2 - markerSize/2, markerSize, markerSize);
            }

            // Draw in-progress bbox
            if (drawing && startPoint && currentPoint) {
                const w = currentPoint.x - startPoint.x;
                const h = currentPoint.y - startPoint.y;

                // Use color based on type
                const color = isNegative ? "#f00" : "#0f0";
                const fillColor = isNegative ? "rgba(255, 0, 0, 0.1)" : "rgba(0, 255, 0, 0.1)";

                ctx.fillStyle = fillColor;
                ctx.fillRect(startPoint.x, startPoint.y, w, h);

                ctx.strokeStyle = color;
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
                    ctx.fillStyle = color;
                    ctx.font = "12px monospace";
                    ctx.textAlign = "center";
                    ctx.fillText(`${Math.round(absW)}x${Math.round(absH)}`, startPoint.x + w/2, startPoint.y + h/2 + 4);
                }
            }
        };

        // Serialize/configure for state persistence
        nodeType.prototype.onSerialize = function(data) {
            if (this._bboxWidget) {
                data.bboxCreatorState = {
                    positive: this._bboxWidget.positiveBBoxes,
                    negative: this._bboxWidget.negativeBBoxes
                };
            }
        };

        const originalConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function(data) {
            if (originalConfigure) {
                originalConfigure.apply(this, arguments);
            }

            if (data.bboxCreatorState) {
                setTimeout(() => {
                    if (this._bboxWidget) {
                        this._bboxWidget.positiveBBoxes = data.bboxCreatorState.positive || [];
                        this._bboxWidget.negativeBBoxes = data.bboxCreatorState.negative || [];
                        this.updateBBoxData();
                        this.redrawCanvas();
                    }
                }, 100);
            }
        };
    }
});

console.log("[NV_BBoxCreator] Extension loaded");
