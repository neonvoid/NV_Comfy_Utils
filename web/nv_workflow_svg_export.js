// NV Workflow SVG Export
// Renders the current ComfyUI graph as a vector SVG using canvas2svg.
//
// Forked from pythongosssss/ComfyUI-Custom-Scripts workflowImage.js (MIT).
// Stripped to SVG-only, renamed to avoid collision, patched for the new
// Comfy V1 frontend (see canvas2svg.js __applyCurrentDefaultPath NV patch).

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

let getDrawTextConfig = null;

class NvWorkflowSvg {
    extension = "svg";

    getBounds() {
        const bounds = app.graph._nodes.reduce(
            (p, n) => {
                if (n.pos[0] < p[0]) p[0] = n.pos[0];
                if (n.pos[1] < p[1]) p[1] = n.pos[1];
                const b = n.getBounding();
                const r = n.pos[0] + b[2];
                const bt = n.pos[1] + b[3];
                if (r > p[2]) p[2] = r;
                if (bt > p[3]) p[3] = bt;
                return p;
            },
            [99999, 99999, -99999, -99999]
        );
        bounds[0] -= 100;
        bounds[1] -= 100;
        bounds[2] += 100;
        bounds[3] += 100;
        return bounds;
    }

    saveState() {
        this.state = {
            scale: app.canvas.ds.scale,
            width: app.canvas.canvas.width,
            height: app.canvas.canvas.height,
            offset: app.canvas.ds.offset,
            transform: app.canvas.canvas.getContext("2d").getTransform(),
            ctx: app.canvas.ctx,
        };
    }

    restoreState() {
        app.canvas.ds.scale = this.state.scale;
        app.canvas.canvas.width = this.state.width;
        app.canvas.canvas.height = this.state.height;
        app.canvas.ds.offset = this.state.offset;
        app.canvas.canvas.getContext("2d").setTransform(this.state.transform);
        app.canvas.ctx = this.state.ctx;
    }

    updateView(bounds) {
        const scale = window.devicePixelRatio || 1;
        app.canvas.ds.scale = 1;
        app.canvas.canvas.width = (bounds[2] - bounds[0]) * scale;
        app.canvas.canvas.height = (bounds[3] - bounds[1]) * scale;
        app.canvas.ds.offset = [-bounds[0], -bounds[1]];
        app.canvas.canvas.getContext("2d").setTransform(scale, 0, 0, scale, 0, 0);
        this.createSvgCtx(bounds);
    }

    createSvgCtx(bounds) {
        const ctx = this.state.ctx;
        const svgCtx = (this.svgCtx = new NV_C2S(bounds[2] - bounds[0], bounds[3] - bounds[1]));
        svgCtx.canvas.getBoundingClientRect = () => ({ width: svgCtx.width, height: svgCtx.height });

        const drawImage = svgCtx.drawImage;
        const debug = window.__NV_SVG_DEBUG === true;
        const diag = { calls: 0, img: 0, video: 0, canvas: 0, skipped: 0, embedded: 0, failed: 0 };
        this._diag = diag;
        svgCtx.drawImage = function (...args) {
            diag.calls++;
            const image = args[0];
            const name = image && image.nodeName;

            // Pre-bake <img>/<video>/<canvas> into a same-origin canvas whose
            // toDataURL we invoke here, so failures (CORS taint, 0-size, unloaded)
            // surface now instead of inside canvas2svg's downstream serializer.
            const bake = (src, w, h) => {
                const c = document.createElement("canvas");
                c.width = Math.max(1, w);
                c.height = Math.max(1, h);
                c.getContext("2d").drawImage(src, 0, 0, c.width, c.height);
                const dataUrl = c.toDataURL("image/png");  // throws on taint
                c.toDataURL = () => dataUrl;  // canvas2svg will call this
                return c;
            };

            try {
                if (name === "VIDEO") {
                    diag.video++;
                    const w = image.videoWidth || image.width;
                    const h = image.videoHeight || image.height;
                    if (!w || !h || image.readyState < 2) {
                        if (debug) console.warn("[NV_WorkflowSvg] video not ready", image.src, { w, h, readyState: image.readyState });
                        diag.skipped++;
                        return;
                    }
                    args[0] = bake(image, w, h);
                    diag.embedded++;
                } else if (name === "IMG") {
                    diag.img++;
                    const w = image.naturalWidth || image.width;
                    const h = image.naturalHeight || image.height;
                    if (!image.complete || !w || !h) {
                        if (debug) console.warn("[NV_WorkflowSvg] img not loaded", image.src, { complete: image.complete, w, h });
                        diag.skipped++;
                        return;
                    }
                    // Bake even data: URLs — canvas2svg's downstream src path
                    // can still break on some SVG consumers, data-URL-in-data-URL is safe.
                    args[0] = bake(image, w, h);
                    diag.embedded++;
                } else if (name === "CANVAS") {
                    diag.canvas++;
                    // Pre-serialize so any taint surfaces here.
                    const dataUrl = image.toDataURL("image/png");
                    const proxy = image;  // canvas2svg will re-call toDataURL
                    const origToDataURL = proxy.toDataURL;
                    proxy.toDataURL = () => dataUrl;
                    // Restore after this draw call to avoid polluting caller's canvas.
                    queueMicrotask(() => { proxy.toDataURL = origToDataURL; });
                    diag.embedded++;
                }
            } catch (err) {
                diag.failed++;
                console.warn("[NV_WorkflowSvg] drawImage: could not embed", name,
                             image && image.src, err && err.message);
                return;
            }
            return drawImage.apply(this, args);
        };

        svgCtx.getTransform = () => ctx.getTransform();
        svgCtx.resetTransform = () => ctx.resetTransform();
        svgCtx.roundRect = svgCtx.rect;
        app.canvas.ctx = svgCtx;
    }

    static escapeXml(s) {
        return s.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
    }

    getDrawTextConfig(_, widget) {
        const domWrapper = widget.inputEl.closest(".dom-widget") ?? widget.inputEl;
        return {
            x: parseInt(domWrapper.style.left),
            y: parseInt(domWrapper.style.top),
            resetTransform: true,
        };
    }

    getBlob(workflow) {
        let svg = this.svgCtx
            .getSerializedSvg(true)
            .replace("<svg ", `<svg style="background: ${app.canvas.clear_background_color}" `);
        if (workflow) {
            svg = svg.replace("</svg>", `<desc>${NvWorkflowSvg.escapeXml(workflow)}</desc></svg>`);
        }
        return new Blob([svg], { type: "image/svg+xml" });
    }

    download(blob) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        Object.assign(a, {
            href: url,
            download: "workflow." + this.extension,
            style: "display: none",
        });
        document.body.append(a);
        a.click();
        setTimeout(() => {
            a.remove();
            URL.revokeObjectURL(url);
        }, 150);
    }

    export(includeWorkflow) {
        this.saveState();
        try {
            this.updateView(this.getBounds());
            getDrawTextConfig = this.getDrawTextConfig;
            app.canvas.draw(true, true);

            const blob = this.getBlob(includeWorkflow ? JSON.stringify(app.graph.serialize()) : undefined);
            console.info("[NV_WorkflowSvg] draw complete", this._diag, "svg bytes:", blob.size);
            this.download(blob);
        } catch (err) {
            console.error("[NV_WorkflowSvg] export failed:", err);
            alert("SVG export failed — see console. (NV_WorkflowSvg)");
        } finally {
            getDrawTextConfig = null;
            this.restoreState();
            app.canvas.draw(true, true);
        }
    }
}

function wrapText(context, text, x, y, maxWidth, lineHeight) {
    const words = text.split(" ");
    let line = "";
    for (let i = 0; i < words.length; i++) {
        let test = words[i];
        let metrics = context.measureText(test);
        while (metrics.width > maxWidth) {
            test = test.substring(0, test.length - 1);
            metrics = context.measureText(test);
        }
        if (words[i] !== test) {
            words.splice(i + 1, 0, words[i].substr(test.length));
            words[i] = test;
        }
        test = line + words[i] + " ";
        metrics = context.measureText(test);
        if (metrics.width > maxWidth && i > 0) {
            context.fillText(line, x, y);
            line = words[i] + " ";
            y += lineHeight;
        } else {
            line = test;
        }
    }
    context.fillText(line, x, y);
}

app.registerExtension({
    name: "nv.WorkflowSvgExport",
    init() {
        if (ComfyWidgets.STRING && ComfyWidgets.STRING.__nv_svg_wrapped) return;
        const stringWidget = ComfyWidgets.STRING;
        const wrapped = function () {
            const w = stringWidget.apply(this, arguments);
            if (w.widget && w.widget.type === "customtext") {
                const draw = w.widget.draw;
                w.widget.draw = function (ctx) {
                    draw.apply(this, arguments);
                    if (this.inputEl.hidden) return;
                    if (getDrawTextConfig) {
                        const config = getDrawTextConfig(ctx, this);
                        const t = ctx.getTransform();
                        ctx.save();
                        if (config.resetTransform) ctx.resetTransform();
                        const style = document.defaultView.getComputedStyle(this.inputEl, null);
                        const x = config.x;
                        const y = config.y;
                        const domWrapper = this.inputEl.closest(".dom-widget") ?? this.inputEl;
                        let w2 = parseInt(domWrapper.style.width, 10);
                        if (!w2) w2 = this.node.size[0] - 20;
                        let h = parseInt(domWrapper.style.height, 10);
                        if (!h) h = this.node.size[1] - 20;
                        ctx.fillStyle = style.getPropertyValue("background-color");
                        ctx.fillRect(x, y, w2, h);
                        ctx.fillStyle = style.getPropertyValue("color");
                        ctx.font = style.getPropertyValue("font");
                        const line = t.d * 12;
                        const split = this.inputEl.value.split("\n");
                        let start = y;
                        for (const l of split) {
                            start += line;
                            wrapText(ctx, l, x + 4, start, w2, line);
                        }
                        ctx.restore();
                    }
                };
            }
            return w;
        };
        wrapped.__nv_svg_wrapped = true;
        ComfyWidgets.STRING = wrapped;
    },
    setup() {
        if (document.getElementById("nv-canvas2svg-script")) return;
        const script = document.createElement("script");
        script.id = "nv-canvas2svg-script";
        script.onerror = (e) => console.error("[NV_WorkflowSvg] Failed to load canvas2svg.js:", e);
        script.onload = function () {
            if (LGraphCanvas.prototype.__nv_svg_menu_patched) return;
            LGraphCanvas.prototype.__nv_svg_menu_patched = true;
            const orig = LGraphCanvas.prototype.getCanvasMenuOptions;
            LGraphCanvas.prototype.getCanvasMenuOptions = function () {
                const options = orig.apply(this, arguments);
                options.push(null, {
                    content: "NV Workflow SVG",
                    submenu: {
                        options: [
                            {
                                content: "Export SVG (with embedded workflow)",
                                callback: () => new NvWorkflowSvg().export(true),
                            },
                            {
                                content: "Export SVG (image only)",
                                callback: () => new NvWorkflowSvg().export(false),
                            },
                        ],
                    },
                });
                return options;
            };
        };
        script.src = new URL("assets/canvas2svg.js", import.meta.url);
        document.body.append(script);
    },
});
