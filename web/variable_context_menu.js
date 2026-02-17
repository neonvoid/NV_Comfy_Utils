/**
 * Variable Context Menu — "Promote to Variable" on node output slots
 *
 * Inspired by Unreal Engine's Blueprint "Promote to Variable" feature.
 * Right-click any node → "Promote to Variable" → select an output slot →
 * choose an existing variable or create a new one.
 *
 * Uses the addMenuHandler pattern from KJNodes/pysssss-custom-scripts.
 */

import { app } from "../../scripts/app.js";
import { variableManager } from "./variable_manager.js";

console.log("[NV_Comfy_Utils] Loading variable context menu...");

function addMenuHandler(nodeType, cb) {
    const getOpts = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function () {
        const r = getOpts.apply(this, arguments);
        cb.apply(this, arguments);
        return r;
    };
}

/**
 * Build the variable selection submenu for a given node + output slot.
 */
function buildVariableOptions(sourceNode, slotIndex) {
    const existingVars = variableManager.getVariableNames();

    const options = existingVars.map(varName => ({
        content: varName,
        callback: () => {
            variableManager.assignSource(varName, sourceNode, slotIndex);
        }
    }));

    // Separator if there are existing vars
    if (options.length > 0) {
        options.push(null); // null = separator in LiteGraph menus
    }

    // "New Variable..." option
    options.push({
        content: "+ New Variable...",
        callback: () => {
            showNewVariableDialog(sourceNode, slotIndex);
        }
    });

    return options;
}

/**
 * Show a dialog to create a new variable and assign the source to it.
 */
function showNewVariableDialog(sourceNode, slotIndex) {
    const overlay = document.createElement("div");
    overlay.style.cssText = `
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.5);
        z-index: 10000;
    `;

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
        z-index: 10001;
        min-width: 300px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.6);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #e0e0e0;
    `;

    const title = document.createElement("h3");
    title.textContent = "New Variable";
    title.style.cssText = "margin: 0 0 8px 0; font-size: 14px;";

    const outputInfo = document.createElement("div");
    const outputName = sourceNode.outputs?.[slotIndex]?.name || `output_${slotIndex}`;
    const outputType = sourceNode.outputs?.[slotIndex]?.type || "*";
    outputInfo.textContent = `Source: ${sourceNode.title || sourceNode.type} → ${outputName} (${outputType})`;
    outputInfo.style.cssText = "margin-bottom: 12px; font-size: 11px; color: #888;";

    const label = document.createElement("label");
    label.textContent = "Variable name:";
    label.style.cssText = "display: block; margin-bottom: 4px; color: #999; font-size: 12px;";

    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "my_variable";
    input.style.cssText = `
        width: 100%;
        padding: 8px;
        margin-bottom: 4px;
        background: #2a2a2a;
        border: 1px solid #444;
        border-radius: 4px;
        color: #e0e0e0;
        box-sizing: border-box;
        font-size: 12px;
        outline: none;
    `;
    input.addEventListener("focus", () => input.style.borderColor = "#5af");
    input.addEventListener("blur", () => input.style.borderColor = "#444");

    const errorMsg = document.createElement("div");
    errorMsg.style.cssText = "color: #f44; font-size: 11px; min-height: 16px; margin-bottom: 8px;";

    const buttons = document.createElement("div");
    buttons.style.cssText = "display: flex; gap: 8px; justify-content: flex-end;";

    const cleanup = () => {
        document.body.removeChild(dialog);
        document.body.removeChild(overlay);
    };

    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancel";
    cancelBtn.style.cssText = `
        padding: 8px 16px;
        background: #333;
        border: none;
        border-radius: 4px;
        color: #e0e0e0;
        cursor: pointer;
        font-size: 12px;
    `;
    cancelBtn.addEventListener("click", cleanup);

    const createBtn = document.createElement("button");
    createBtn.textContent = "Create & Assign";
    createBtn.style.cssText = `
        padding: 8px 16px;
        background: #4ade80;
        border: none;
        border-radius: 4px;
        color: #111;
        cursor: pointer;
        font-size: 12px;
        font-weight: 500;
    `;
    createBtn.addEventListener("mouseenter", () => createBtn.style.filter = "brightness(1.1)");
    createBtn.addEventListener("mouseleave", () => createBtn.style.filter = "none");

    const doCreate = () => {
        const name = input.value.trim();
        if (!name) {
            errorMsg.textContent = "Name cannot be empty";
            return;
        }

        // Check for existing variable
        const existingNames = variableManager.getVariableNames();
        if (existingNames.includes(name)) {
            errorMsg.textContent = `Variable "${name}" already exists`;
            return;
        }

        const setter = variableManager.createVariable(name);
        if (setter) {
            variableManager.assignSource(name, sourceNode, slotIndex);
        }
        cleanup();
    };

    createBtn.addEventListener("click", doCreate);

    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            doCreate();
        } else if (e.key === "Escape") {
            cleanup();
        }
    });

    // Clear error on input
    input.addEventListener("input", () => {
        errorMsg.textContent = "";
    });

    overlay.addEventListener("click", cleanup);

    buttons.appendChild(cancelBtn);
    buttons.appendChild(createBtn);

    dialog.appendChild(title);
    dialog.appendChild(outputInfo);
    dialog.appendChild(label);
    dialog.appendChild(input);
    dialog.appendChild(errorMsg);
    dialog.appendChild(buttons);

    document.body.appendChild(overlay);
    document.body.appendChild(dialog);

    // Focus and auto-suggest a name based on output
    const suggestedName = (outputName !== "*" && outputName !== `output_${slotIndex}`)
        ? outputName.toLowerCase().replace(/\s+/g, "_")
        : "";
    input.value = suggestedName;
    input.focus();
    input.select();
}

// Register the context menu extension
app.registerExtension({
    name: "NV_Comfy_Utils.VariableContextMenu",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only add to nodes that have a Python backend definition (real nodes)
        if (!nodeData?.input) return;

        addMenuHandler(nodeType, function (_, options) {
            // Don't add to variable nodes themselves
            if (this.type === "SetVariableNode" || this.type === "GetVariableNode") return;

            // Must have outputs
            if (!this.outputs || this.outputs.length === 0) return;

            const node = this;

            if (node.outputs.length === 1) {
                // Single output — flatten the menu (no slot picker needed)
                const output = node.outputs[0];
                const outputLabel = output.name || output.type || "output";

                options.push(null); // separator
                options.push({
                    content: `Promote to Variable (${outputLabel})`,
                    has_submenu: true,
                    submenu: {
                        title: "Assign to Variable",
                        options: buildVariableOptions(node, 0),
                    }
                });
            } else {
                // Multiple outputs — show slot picker first
                const outputSubmenus = node.outputs.map((output, slotIdx) => {
                    const outputLabel = output.name || output.type || `output_${slotIdx}`;
                    return {
                        content: `${outputLabel} (${output.type})`,
                        has_submenu: true,
                        submenu: {
                            title: "Assign to Variable",
                            options: buildVariableOptions(node, slotIdx),
                        }
                    };
                });

                options.push(null); // separator
                options.push({
                    content: "Promote to Variable",
                    has_submenu: true,
                    submenu: {
                        title: "Select Output",
                        options: outputSubmenus,
                    }
                });
            }
        });
    }
});

console.log("[NV_Comfy_Utils] Variable context menu loaded");
