import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

/**
 * NV Frame Annotator Frontend Extension
 * Adds frame scrubbing and marking UI to the NV_FrameAnnotator node
 */

app.registerExtension({
    name: "NV_Comfy_Utils.FrameAnnotator",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "NV_FrameAnnotator") {
            return;
        }

        console.log("[FrameAnnotator] Registering extension for NV_FrameAnnotator");

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function() {
            // Call original if exists
            if (originalOnNodeCreated) {
                originalOnNodeCreated.apply(this, arguments);
            }

            const node = this;

            // Initialize internal state
            node._frameAnnotator = {
                currentFrame: 0,
                totalFrames: 0,
                markedFrames: new Set(),
                initialized: false
            };

            // Find the marked_frames widget (added by Python node)
            const markedFramesWidget = node.widgets?.find(w => w.name === "marked_frames");

            // Parse existing marked frames from widget
            if (markedFramesWidget && markedFramesWidget.value) {
                const existing = markedFramesWidget.value.split(",")
                    .map(s => s.trim())
                    .filter(s => s && !isNaN(parseInt(s)))
                    .map(s => parseInt(s));
                node._frameAnnotator.markedFrames = new Set(existing);
            }

            // === Frame Counter Display ===
            const frameCounterWidget = ComfyWidgets["STRING"](node, "frame_display", ["STRING", {
                default: "Frame: 0 / ?",
                multiline: false
            }], app).widget;
            frameCounterWidget.name = "Frame Display";
            frameCounterWidget.inputEl.readOnly = true;
            frameCounterWidget.inputEl.style.textAlign = "center";
            frameCounterWidget.inputEl.style.fontWeight = "bold";

            // === Frame Scrubber Slider ===
            const scrubberWidget = node.addWidget("slider", "frame_scrubber", 0, (value) => {
                node._frameAnnotator.currentFrame = Math.floor(value);
                updateDisplay(node);
            }, {
                min: 0,
                max: 100,
                step: 1
            });
            scrubberWidget.name = "Frame";

            // === Navigation Buttons Row ===
            // Previous Frame Button
            const prevButton = node.addWidget("button", "prev_frame", "< Prev", () => {
                navigateFrame(node, -1);
            });
            prevButton.serialize = false;

            // Next Frame Button
            const nextButton = node.addWidget("button", "next_frame", "Next >", () => {
                navigateFrame(node, 1);
            });
            nextButton.serialize = false;

            // === Mark Frame Button ===
            const markButton = node.addWidget("button", "mark_frame", "Mark Frame", () => {
                toggleMarkFrame(node);
            });
            markButton.serialize = false;
            node._markButton = markButton;

            // === Marked Frames Display ===
            const markedDisplayWidget = ComfyWidgets["STRING"](node, "marked_display", ["STRING", {
                default: "Marked: (none)",
                multiline: true
            }], app).widget;
            markedDisplayWidget.name = "Marked Frames";
            markedDisplayWidget.inputEl.readOnly = true;
            markedDisplayWidget.inputEl.style.fontSize = "11px";

            // Store widget references
            node._frameCounterWidget = frameCounterWidget;
            node._scrubberWidget = scrubberWidget;
            node._markedDisplayWidget = markedDisplayWidget;

            // Override serialize to persist state
            const originalSerialize = node.serialize;
            node.serialize = function() {
                const data = originalSerialize ? originalSerialize.call(this) : {};
                data.frameAnnotatorState = {
                    currentFrame: this._frameAnnotator.currentFrame,
                    markedFrames: Array.from(this._frameAnnotator.markedFrames)
                };
                return data;
            };

            // Override configure to restore state
            const originalConfigure = node.configure;
            node.configure = function(data) {
                if (originalConfigure) {
                    originalConfigure.call(this, data);
                }

                if (data.frameAnnotatorState) {
                    this._frameAnnotator.currentFrame = data.frameAnnotatorState.currentFrame || 0;
                    this._frameAnnotator.markedFrames = new Set(data.frameAnnotatorState.markedFrames || []);

                    // Sync to widget
                    syncMarkedFramesToWidget(this);
                    updateDisplay(this);
                }
            };

            // Set initial size
            node.size = [300, 320];

            // Initial display update
            updateDisplay(node);

            console.log("[FrameAnnotator] Node initialized successfully");
        };
    },

    // Handle when node executes and we get frame count
    async nodeCreated(node) {
        if (node.comfyClass !== "NV_FrameAnnotator") {
            return;
        }

        // Poll for execution results to get frame count
        const checkForResults = () => {
            if (node.widgets) {
                // After execution, the output frame_count will be available
                // We can also check if images input is connected
                updateTotalFramesFromConnection(node);
            }
        };

        // Set up periodic check for frame count updates
        node._frameCheckInterval = setInterval(checkForResults, 1000);

        // Clean up on removal
        const originalOnRemoved = node.onRemoved;
        node.onRemoved = function() {
            if (this._frameCheckInterval) {
                clearInterval(this._frameCheckInterval);
            }
            if (originalOnRemoved) {
                originalOnRemoved.call(this);
            }
        };
    }
});

// === Helper Functions ===

function navigateFrame(node, delta) {
    const state = node._frameAnnotator;
    const maxFrame = state.totalFrames > 0 ? state.totalFrames - 1 : 100;

    state.currentFrame = Math.max(0, Math.min(maxFrame, state.currentFrame + delta));

    // Update scrubber
    if (node._scrubberWidget) {
        node._scrubberWidget.value = state.currentFrame;
    }

    updateDisplay(node);

    // Visual feedback
    flashNode(node, delta > 0 ? "#4a6a8a" : "#6a4a8a");
}

function toggleMarkFrame(node) {
    const state = node._frameAnnotator;
    const currentFrame = state.currentFrame;

    if (state.markedFrames.has(currentFrame)) {
        state.markedFrames.delete(currentFrame);
        console.log(`[FrameAnnotator] Unmarked frame ${currentFrame}`);
    } else {
        state.markedFrames.add(currentFrame);
        console.log(`[FrameAnnotator] Marked frame ${currentFrame}`);
    }

    // Sync to Python widget
    syncMarkedFramesToWidget(node);
    updateDisplay(node);

    // Visual feedback
    const isMarked = state.markedFrames.has(currentFrame);
    flashNode(node, isMarked ? "#4a8a4a" : "#8a4a4a");
}

function syncMarkedFramesToWidget(node) {
    // Find the marked_frames widget from Python node
    const markedFramesWidget = node.widgets?.find(w => w.name === "marked_frames");
    if (markedFramesWidget) {
        const sortedFrames = Array.from(node._frameAnnotator.markedFrames).sort((a, b) => a - b);
        markedFramesWidget.value = sortedFrames.join(",");
    }
}

function updateDisplay(node) {
    const state = node._frameAnnotator;

    // Update frame counter
    if (node._frameCounterWidget) {
        const totalStr = state.totalFrames > 0 ? state.totalFrames - 1 : "?";
        node._frameCounterWidget.value = `Frame: ${state.currentFrame} / ${totalStr}`;
    }

    // Update mark button text
    if (node._markButton) {
        const isMarked = state.markedFrames.has(state.currentFrame);
        node._markButton.name = isMarked ? "★ Unmark Frame" : "Mark Frame";
    }

    // Update marked frames display
    if (node._markedDisplayWidget) {
        const sortedFrames = Array.from(state.markedFrames).sort((a, b) => a - b);
        let displayText;

        if (sortedFrames.length === 0) {
            displayText = "Marked: (none)";
        } else if (sortedFrames.length <= 15) {
            displayText = `Marked (${sortedFrames.length}): ${sortedFrames.join(", ")}`;
        } else {
            // Show condensed version
            const first = sortedFrames.slice(0, 5).join(", ");
            const last = sortedFrames.slice(-3).join(", ");
            displayText = `Marked (${sortedFrames.length}): ${first} ... ${last}`;
        }

        node._markedDisplayWidget.value = displayText;
    }

    // Update scrubber max
    if (node._scrubberWidget && state.totalFrames > 0) {
        node._scrubberWidget.options.max = state.totalFrames - 1;
    }

    // Trigger redraw
    if (node.setDirtyCanvas) {
        node.setDirtyCanvas(true, true);
    }
}

function updateTotalFramesFromConnection(node) {
    // Try to get frame count from connected node's output
    const imagesInput = node.inputs?.find(i => i.name === "images");
    if (!imagesInput || imagesInput.link === null) {
        return;
    }

    const link = node.graph?.links?.[imagesInput.link];
    if (!link) {
        return;
    }

    const originNode = node.graph?.getNodeById?.(link.origin_id);
    if (!originNode) {
        return;
    }

    // Try to find frame count from origin node
    // Check if origin has a known output property
    let frameCount = null;

    // Check for various common frame count sources
    if (originNode.widgets) {
        const totalFramesWidget = originNode.widgets.find(w =>
            w.name === "frame_count" ||
            w.name === "total_frames" ||
            w.name === "frames"
        );
        if (totalFramesWidget && totalFramesWidget.value > 0) {
            frameCount = totalFramesWidget.value;
        }
    }

    // If we found a frame count and it changed, update
    if (frameCount && frameCount !== node._frameAnnotator.totalFrames) {
        node._frameAnnotator.totalFrames = frameCount;

        // Update scrubber max
        if (node._scrubberWidget) {
            node._scrubberWidget.options.max = frameCount - 1;
            // Clamp current frame
            if (node._frameAnnotator.currentFrame >= frameCount) {
                node._frameAnnotator.currentFrame = frameCount - 1;
                node._scrubberWidget.value = node._frameAnnotator.currentFrame;
            }
        }

        updateDisplay(node);
        console.log(`[FrameAnnotator] Updated total frames to ${frameCount}`);
    }
}

function flashNode(node, color) {
    const originalBgColor = node.bgcolor;
    node.bgcolor = color;

    if (node.setDirtyCanvas) {
        node.setDirtyCanvas(true, true);
    }

    setTimeout(() => {
        node.bgcolor = originalBgColor;
        if (node.setDirtyCanvas) {
            node.setDirtyCanvas(true, true);
        }
    }, 150);
}

console.log("[NV_Comfy_Utils] Frame Annotator extension loaded");
