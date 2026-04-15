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
        svgCtx.drawImage = function (...args) {
            const image = args[0];
            if (image.nodeName === "IMG" && !image.src.startsWith("data:image/")) {
                const canvas = document.createElement("canvas");
                canvas.width = image.width;
                canvas.height = image.height;
                canvas.getContext("2d").drawImage(image, 0, 0);
                args[0] = canvas;
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
