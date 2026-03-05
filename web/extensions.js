// Import extensions sequentially (order matters for dependencies)
// Dynamic imports isolate failures so one broken module doesn't kill all others
console.log("[NV_Comfy_Utils] Loading extensions...");

const modules = [
    "./node_bypasser.js",
    "./stable_naming.js",
    "./variable_manager.js",        // Must load before simple_variables, variable_context_menu, variables_panel
    "./simple_variables.js",
    "./variable_context_menu.js",
    "./simple_link_switcher.js",
    "./momentary_button.js",
    "./frame_annotator.js",
    "./bbox_creator.js",
    "./floating_panel.js",          // Must load before variables_panel
    "./variables_panel.js",
    "./download_video.js",
    "./preview_animation.js",
    "./clone_with_connections.js",
    "./point_picker.js",
];

for (const mod of modules) {
    try {
        await import(mod);
    } catch (err) {
        console.error(`[NV_Comfy_Utils] Failed to load ${mod}:`, err);
    }
}

console.log("[NV_Comfy_Utils] All extensions loaded successfully");
