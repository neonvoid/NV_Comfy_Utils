/**
 * NV Preview Animation - Fast flipbook player widget.
 * Receives individual JPEG frames from the Python node and animates them
 * on a canvas with play/pause, scrub, step, and speed controls.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("[NV_PreviewAnimation] Loading extension...");

/**
 * Build a ComfyUI /view URL for a frame image result.
 */
function frameUrl(img) {
    const params = new URLSearchParams({
        filename: img.filename,
        subfolder: img.subfolder || "",
        type: img.type || "temp",
    });
    return api.apiURL("/view?" + params.toString());
}

/**
 * Draw a single frame onto the player canvas, scaled to fit.
 */
function drawFrame(player, index) {
    const img = player.frames[index];
    if (!img) return;
    const { ctx, canvas } = player;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
}

/**
 * Update the frame counter label.
 */
function updateLabels(player) {
    const loaded = player.loaded;
    const total = player.totalFrames;
    const cur = player.currentFrame + 1;
    if (loaded < total) {
        player.frameLabel.textContent = `Loading ${loaded}/${total}`;
    } else {
        player.frameLabel.textContent = `${cur} / ${total}`;
    }
    player.fpsLabel.textContent = `${player.fps.toFixed(1)} fps`;
}

/**
 * Start the animation loop.
 */
function startPlayback(player) {
    if (player.totalFrames < 2) return;
    player.playing = true;
    player.playBtn.textContent = "\u23F8";  // pause symbol
    player.lastFrameTime = performance.now();
    animate(player);
}

/**
 * Stop playback.
 */
function stopPlayback(player) {
    player.playing = false;
    player.playBtn.textContent = "\u25B6";  // play symbol
    if (player.animId) {
        cancelAnimationFrame(player.animId);
        player.animId = null;
    }
}

/**
 * requestAnimationFrame loop with time-based frame advancement.
 */
function animate(player) {
    if (!player.playing) return;

    const now = performance.now();
    const frameDuration = 1000 / player.fps;
    const elapsed = now - player.lastFrameTime;

    if (elapsed >= frameDuration) {
        const framesToAdvance = Math.max(1, Math.floor(elapsed / frameDuration));
        player.currentFrame = (player.currentFrame + framesToAdvance) % player.totalFrames;
        player.lastFrameTime = now - (elapsed % frameDuration);

        drawFrame(player, player.currentFrame);
        player.scrubber.value = player.currentFrame;
        updateLabels(player);
    }

    player.animId = requestAnimationFrame(() => animate(player));
}

/**
 * Resize the node to fit the canvas aspect ratio + controls.
 */
function resizeNodeToFit(node, player) {
    const nodeWidth = node.size[0];
    const controlsHeight = 56;
    if (player.imageWidth > 0 && player.imageHeight > 0) {
        const aspectRatio = player.imageHeight / player.imageWidth;
        const canvasHeight = nodeWidth * aspectRatio;
        player.domWidget.computeSize = (width) => [width, canvasHeight + controlsHeight];
    }
    node.setDirtyCanvas(true, true);
}

/**
 * Create a small styled button.
 */
function makeButton(text, title) {
    const btn = document.createElement("button");
    btn.textContent = text;
    btn.title = title || "";
    btn.style.cssText = `
        background: #444; color: #eee; border: 1px solid #666; border-radius: 3px;
        padding: 2px 8px; cursor: pointer; font-size: 14px; line-height: 1.2;
        font-family: monospace; min-width: 28px; text-align: center;
    `;
    btn.addEventListener("mouseenter", () => { btn.style.background = "#555"; });
    btn.addEventListener("mouseleave", () => { btn.style.background = "#444"; });
    return btn;
}


/**
 * Encode all loaded frames into a WebM video blob using MediaRecorder.
 * Uses captureStream(0) for manual frame pushing.
 * Note: MediaRecorder records wall-clock time between requestFrame() calls,
 * so we use the actual frameDuration to get correct playback speed.
 */
async function exportToWebM(player, progressCb) {
    const { frames, totalFrames, fps, imageWidth, imageHeight } = player;
    console.log(`[NV_PreviewAnimation] exportToWebM: ${totalFrames} frames, ${fps} fps, ${imageWidth}x${imageHeight}`);

    if (!totalFrames || !frames[0]) {
        console.warn("[NV_PreviewAnimation] exportToWebM: no frames loaded");
        return null;
    }

    const offscreen = document.createElement("canvas");
    offscreen.width = imageWidth;
    offscreen.height = imageHeight;
    const offCtx = offscreen.getContext("2d");

    const stream = offscreen.captureStream(0);
    const videoTrack = stream.getVideoTracks()[0];
    console.log("[NV_PreviewAnimation] captureStream created, videoTrack:", !!videoTrack);

    if (!videoTrack) {
        console.error("[NV_PreviewAnimation] No video track from captureStream — browser may not support this");
        return null;
    }

    let mimeType = "video/webm; codecs=vp9";
    if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = "video/webm; codecs=vp8";
        if (!MediaRecorder.isTypeSupported(mimeType)) {
            mimeType = "video/webm";
        }
    }
    console.log("[NV_PreviewAnimation] Using mimeType:", mimeType);

    const recorder = new MediaRecorder(stream, {
        mimeType,
        videoBitsPerSecond: 8_000_000,
    });

    const chunks = [];
    recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
        console.log(`[NV_PreviewAnimation] ondataavailable: chunk ${chunks.length}, size=${e.data.size}`);
    };

    recorder.onerror = (e) => {
        console.error("[NV_PreviewAnimation] MediaRecorder error:", e);
    };

    const frameDuration = 1000 / fps;
    console.log(`[NV_PreviewAnimation] frameDuration=${frameDuration.toFixed(1)}ms, total encode time ~${((totalFrames * frameDuration) / 1000).toFixed(1)}s`);

    return new Promise((resolve) => {
        recorder.onstop = () => {
            const blob = new Blob(chunks, { type: "video/webm" });
            console.log(`[NV_PreviewAnimation] Encoding complete: ${chunks.length} chunks, blob size=${(blob.size / 1024).toFixed(1)}KB`);
            resolve(blob);
        };

        recorder.start(100); // request data every 100ms for progress visibility
        let i = 0;

        function nextFrame() {
            if (i >= totalFrames) {
                console.log("[NV_PreviewAnimation] All frames pushed, stopping recorder...");
                recorder.stop();
                return;
            }
            const img = frames[i];
            if (img) {
                offCtx.drawImage(img, 0, 0, offscreen.width, offscreen.height);
                videoTrack.requestFrame();
            } else {
                console.warn(`[NV_PreviewAnimation] Frame ${i} is null/missing, skipping`);
            }

            if (progressCb) progressCb(i + 1, totalFrames);
            i++;
            setTimeout(nextFrame, frameDuration);
        }

        console.log("[NV_PreviewAnimation] Starting frame encoding loop...");
        nextFrame();
    });
}

/**
 * Trigger a browser file download from a Blob.
 */
function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.setAttribute("download", filename);
    document.body.append(a);
    a.click();
    requestAnimationFrame(() => {
        a.remove();
        URL.revokeObjectURL(url);
    });
}


app.registerExtension({
    name: "NV_Comfy_Utils.PreviewAnimation",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== "NV_PreviewAnimation") return;

        console.log("[NV_PreviewAnimation] Registering extension");

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;

            // ── Container ──
            const container = document.createElement("div");
            container.style.cssText = `
                position: relative; width: 100%; background: #1a1a1a;
                overflow: hidden; box-sizing: border-box;
            `;

            // ── Canvas ──
            const canvas = document.createElement("canvas");
            canvas.width = 512;
            canvas.height = 512;
            canvas.style.cssText = `
                display: block; width: 100%; background: #111;
                image-rendering: auto; cursor: pointer;
            `;
            container.appendChild(canvas);
            const ctx = canvas.getContext("2d");

            // ── Controls bar ──
            const controls = document.createElement("div");
            controls.style.cssText = `
                display: flex; align-items: center; gap: 4px;
                padding: 4px 6px; background: #2a2a2a; user-select: none;
            `;

            const playBtn = makeButton("\u25B6", "Play / Pause");
            const prevBtn = makeButton("\u23EA", "Previous frame");
            const nextBtn = makeButton("\u23E9", "Next frame");

            const frameLabel = document.createElement("span");
            frameLabel.style.cssText = `
                font-size: 11px; font-family: monospace; color: #aaa;
                min-width: 70px; text-align: center;
            `;
            frameLabel.textContent = "0 / 0";

            const fpsLabel = document.createElement("span");
            fpsLabel.style.cssText = `
                font-size: 11px; font-family: monospace; color: #8af;
                min-width: 55px; text-align: right; cursor: ns-resize;
            `;
            fpsLabel.title = "Scroll to adjust FPS";
            fpsLabel.textContent = "8.0 fps";

            controls.appendChild(prevBtn);
            controls.appendChild(playBtn);
            controls.appendChild(nextBtn);
            controls.appendChild(frameLabel);

            // Spacer to push FPS label to the right
            const spacer = document.createElement("div");
            spacer.style.flex = "1";
            controls.appendChild(spacer);
            controls.appendChild(fpsLabel);

            container.appendChild(controls);

            // ── Scrubber ──
            const scrubberRow = document.createElement("div");
            scrubberRow.style.cssText = `
                padding: 2px 6px 4px 6px; background: #2a2a2a;
            `;
            const scrubber = document.createElement("input");
            scrubber.type = "range";
            scrubber.min = 0;
            scrubber.max = 0;
            scrubber.value = 0;
            scrubber.style.cssText = `width: 100%; margin: 0; cursor: pointer;`;
            scrubberRow.appendChild(scrubber);
            container.appendChild(scrubberRow);

            // ── DOM Widget ──
            const widget = node.addDOMWidget("preview_anim", "previewAnimCanvas", container);
            widget.computeSize = (width) => [width, 340];

            // ── Player state ──
            node._animPlayer = {
                canvas, ctx, container,
                frames: [],
                currentFrame: 0,
                totalFrames: 0,
                fps: 8.0,
                playing: false,
                lastFrameTime: 0,
                animId: null,
                loaded: 0,
                imageWidth: 0,
                imageHeight: 0,
                playBtn, prevBtn, nextBtn, frameLabel, fpsLabel, scrubber,
                domWidget: widget,
            };

            // ── Event handlers ──

            // Stop propagation helper to prevent LiteGraph from stealing events
            const stopProp = (e) => { e.stopPropagation(); };

            // Play / Pause
            playBtn.addEventListener("click", (e) => {
                stopProp(e);
                const p = node._animPlayer;
                if (p.playing) {
                    stopPlayback(p);
                } else {
                    startPlayback(p);
                }
            });

            // Canvas click toggles play
            canvas.addEventListener("click", (e) => {
                stopProp(e);
                const p = node._animPlayer;
                if (p.playing) {
                    stopPlayback(p);
                } else {
                    startPlayback(p);
                }
            });

            // Prev frame
            prevBtn.addEventListener("click", (e) => {
                stopProp(e);
                const p = node._animPlayer;
                stopPlayback(p);
                if (p.totalFrames > 0) {
                    p.currentFrame = (p.currentFrame - 1 + p.totalFrames) % p.totalFrames;
                    drawFrame(p, p.currentFrame);
                    p.scrubber.value = p.currentFrame;
                    updateLabels(p);
                }
            });

            // Next frame
            nextBtn.addEventListener("click", (e) => {
                stopProp(e);
                const p = node._animPlayer;
                stopPlayback(p);
                if (p.totalFrames > 0) {
                    p.currentFrame = (p.currentFrame + 1) % p.totalFrames;
                    drawFrame(p, p.currentFrame);
                    p.scrubber.value = p.currentFrame;
                    updateLabels(p);
                }
            });

            // Scrubber
            scrubber.addEventListener("input", (e) => {
                stopProp(e);
                const p = node._animPlayer;
                stopPlayback(p);
                p.currentFrame = parseInt(scrubber.value);
                drawFrame(p, p.currentFrame);
                updateLabels(p);
            });
            scrubber.addEventListener("mousedown", stopProp);
            scrubber.addEventListener("pointerdown", stopProp);

            // FPS adjustment via scroll wheel on the FPS label
            fpsLabel.addEventListener("wheel", (e) => {
                e.preventDefault();
                stopProp(e);
                const p = node._animPlayer;
                const delta = e.deltaY > 0 ? -0.5 : 0.5;
                p.fps = Math.max(0.5, Math.min(120, p.fps + delta));
                updateLabels(p);
            });

            // Prevent all pointer events on controls from bubbling to LiteGraph
            container.addEventListener("pointerdown", stopProp);
            container.addEventListener("pointermove", stopProp);
            container.addEventListener("pointerup", stopProp);
            container.addEventListener("wheel", (e) => {
                // Only prevent default on the FPS label area, let other scrolls pass
                if (e.target === fpsLabel) {
                    e.preventDefault();
                }
                stopProp(e);
            });

            // Initial node size
            node.setSize([350, 460]);

            return result;
        };

        // ── onExecuted: receive frames from Python ──
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            const player = this._animPlayer;
            if (!player) return;
            if (!message?.frames?.length) return;

            // Stop any existing playback
            stopPlayback(player);

            // Read metadata
            const fps = message.fps?.[0] ?? 8.0;
            const frameCount = message.frame_count?.[0] ?? message.frames.length;

            player.fps = fps;
            player.totalFrames = frameCount;
            player.currentFrame = 0;
            player.loaded = 0;

            // Update scrubber range
            player.scrubber.max = Math.max(0, frameCount - 1);
            player.scrubber.value = 0;
            updateLabels(player);

            // Preload all frames
            player.frames = new Array(frameCount).fill(null);

            message.frames.forEach((imgData, i) => {
                const img = new Image();
                img.onload = () => {
                    player.frames[i] = img;
                    player.loaded++;

                    // First frame: set canvas dimensions and display
                    if (i === 0) {
                        player.imageWidth = img.naturalWidth;
                        player.imageHeight = img.naturalHeight;
                        player.canvas.width = img.naturalWidth;
                        player.canvas.height = img.naturalHeight;
                        drawFrame(player, 0);
                        resizeNodeToFit(this, player);
                    }

                    updateLabels(player);

                    // Auto-start when all frames are loaded
                    if (player.loaded === player.totalFrames) {
                        startPlayback(player);
                    }
                };
                img.src = frameUrl(imgData);
            });
        };

        // ── Right-click context menu ──
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            getExtraMenuOptions?.apply(this, arguments);

            const player = this._animPlayer;
            if (!player || !player.totalFrames) return;

            options.unshift(
                {
                    content: `Save as WebM Video (${player.totalFrames} frames)`,
                    callback: async () => {
                        const wasPlaying = player.playing;
                        if (wasPlaying) stopPlayback(player);

                        console.log("[NV_PreviewAnimation] Save as WebM clicked");
                        const startTime = performance.now();

                        const blob = await exportToWebM(player, (done, total) => {
                            player.frameLabel.textContent = `Encoding ${done}/${total}...`;
                        });

                        const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
                        console.log(`[NV_PreviewAnimation] Export finished in ${elapsed}s, blob=${blob ? (blob.size / 1024).toFixed(1) + 'KB' : 'null'}`);

                        if (blob) {
                            const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
                            downloadBlob(blob, `NV_preview_${timestamp}.webm`);
                        }

                        updateLabels(player);
                        if (wasPlaying) startPlayback(player);
                    },
                },
                {
                    content: "Save Current Frame",
                    callback: () => {
                        const { canvas: c, currentFrame } = player;
                        c.toBlob((blob) => {
                            if (blob) {
                                downloadBlob(blob, `NV_frame_${currentFrame}.png`);
                            }
                        }, "image/png");
                    },
                },
                {
                    content: "Open Current Frame in New Tab",
                    callback: () => {
                        const dataUrl = player.canvas.toDataURL("image/png");
                        window.open(dataUrl, "_blank");
                    },
                },
                null // separator
            );
        };

        // ── Cleanup on node removal ──
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            const player = this._animPlayer;
            if (player) {
                stopPlayback(player);
                player.frames = [];
            }
            onRemoved?.apply(this, arguments);
        };
    },
});

console.log("[NV_PreviewAnimation] Extension loaded successfully");
