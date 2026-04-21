import { app } from "../../scripts/app.js";

// Clone With Connections — Shift+Alt+Drag to clone a node while preserving external input
// connections. Normal Alt+Drag behavior (clone without connections) is unchanged.
//
// Strategy: patch LGraphCanvas.prototype._deserializeItems, the chokepoint BOTH clone paths
// go through (classic canvas Alt-drag AND Vue-mode cloneNodes). When shift is held, we
// enable the native `connectInputs` option and its gating setting, which tells the
// deserializer to fall back to graph.getNodeById() for origin_ids that weren't in the
// cloned selection. No manual connect() calls needed — we reuse native infrastructure,
// which means the restored connections live inside the same beforeChange/afterChange
// undo unit as the clone itself (single Ctrl+Z fully undoes).

const LOG_PREFIX = "[CloneWithConnections]";

// Track shift independently — event.shiftKey isn't reliable on every event source,
// and menu-triggered clones have no keyboard event at all.
let shiftHeld = false;
document.addEventListener("keydown", (e) => { if (e.key === "Shift") shiftHeld = true; }, true);
document.addEventListener("keyup", (e) => { if (e.key === "Shift") shiftHeld = false; }, true);
window.addEventListener("blur", () => { shiftHeld = false; });
// Resync on every pointerdown — focused inputs can swallow keyup and leave shiftHeld stuck.
document.addEventListener("pointerdown", (e) => { shiftHeld = e.shiftKey; }, true);

let firstCallLogged = false;

function patchDeserializeItems(LGC) {
    if (!LGC?.prototype || typeof LGC.prototype._deserializeItems !== "function") return false;
    if (LGC.prototype.__nvCloneWithConnectionsPatched) return true;

    const orig = LGC.prototype._deserializeItems;
    const Lite = window.LiteGraph || window.Lite || null;

    LGC.prototype._deserializeItems = function (data, opts = {}) {
        if (!firstCallLogged) {
            firstCallLogged = true;
            console.warn(LOG_PREFIX, `_deserializeItems intercepted (first call). shiftHeld=${shiftHeld}`);
        }
        if (!shiftHeld) return orig.call(this, data, opts);

        // Native gating: the deserializer only falls back to external node lookup when
        // BOTH connectInputs is true AND the setting is enabled. Force both, restore after.
        const prevSetting = Lite?.ctrl_shift_v_paste_connect_unselected_outputs;
        if (Lite) Lite.ctrl_shift_v_paste_connect_unselected_outputs = true;
        try {
            const result = orig.call(this, data, { ...opts, connectInputs: true });
            const created = result?.created?.length ?? 0;
            if (created > 0) console.warn(LOG_PREFIX, `Shift-clone: restored external inputs for ${created} item(s)`);
            return result;
        } finally {
            if (Lite && prevSetting !== undefined) {
                Lite.ctrl_shift_v_paste_connect_unselected_outputs = prevSetting;
            }
        }
    };

    LGC.prototype.__nvCloneWithConnectionsPatched = true;
    return true;
}

function resolveLGraphCanvas() {
    return (
        app.canvas?.constructor ||
        window.LGraphCanvas ||
        window.LiteGraph?.LGraphCanvas ||
        null
    );
}

app.registerExtension({
    name: "NV_Comfy_Utils.CloneWithConnections",

    setup() {
        const tryPatch = () => {
            const LGC = resolveLGraphCanvas();
            if (patchDeserializeItems(LGC)) {
                console.warn(LOG_PREFIX, "Shift+Alt+Drag clone-with-connections ready");
                window.__nvCloneDiag = () => ({
                    patched: LGC.prototype.__nvCloneWithConnectionsPatched === true,
                    shiftHeld,
                    canvasCtorName: app.canvas?.constructor?.name,
                    liteGraphSettingAvailable: typeof (window.LiteGraph?.ctrl_shift_v_paste_connect_unselected_outputs) !== "undefined",
                    LGCSource: LGC === app.canvas?.constructor ? "app.canvas.constructor"
                        : LGC === window.LGraphCanvas ? "window.LGraphCanvas"
                        : LGC === window.LiteGraph?.LGraphCanvas ? "window.LiteGraph.LGraphCanvas"
                        : "unknown",
                });
                return true;
            }
            return false;
        };

        if (tryPatch()) return;

        console.log(LOG_PREFIX, "Canvas not ready, deferring patch...");
        let tries = 0;
        const iv = setInterval(() => {
            tries++;
            if (tryPatch() || tries > 120) {
                clearInterval(iv);
                if (tries > 120) {
                    console.warn(
                        LOG_PREFIX,
                        "Gave up after 2s — LGraphCanvas.prototype._deserializeItems not found.",
                        "Shift+Alt+Drag clone-with-connections is DISABLED for this session.",
                        "Likely cause: ComfyUI frontend refactored the deserialize path — check the bundle for the new method name."
                    );
                }
            }
        }, 16);
    },
});
