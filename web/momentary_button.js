import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Momentary Button - Outputs INT (incrementing) or BOOLEAN (momentary pulse) on each button press
class MomentaryButton extends LGraphNode {
    constructor(title = "Momentary Button") {
        super(title);
        this.type = "NV/MomentaryButton";  // Explicitly set type for proper serialization
        this.comfyClass = "MomentaryButton";
        this.isVirtualNode = true;
        this.removed = false;
        this.configuring = false;
        this._tempWidth = 0;
        this.__constructed__ = false;
        this.widgets = this.widgets || [];
        this.properties = this.properties || {};
        
        console.log("[MomentaryButton] Constructor called");
        
        // Initialize size first to prevent errors
        this.size = [200, 130];
        
        // Internal state
        this._triggerValue = 0;  // For increment mode
        this._pulseState = false;  // For pulse mode (always returns to false)
        this._pulseTimeout = null;  // Timeout handle for auto-reset
        
        // Add mode selector FIRST
        this.modeWidget = this.addWidget("combo", "mode", "Increment", (value) => {
            this.onModeChange(value);
        }, { values: ["Increment", "Pulse"] });
        this.modeWidget.name = "Mode";
        
        // Add pulse duration widget (only visible in Pulse mode)
        this.pulseDurationWidget = ComfyWidgets["INT"](this, "pulse_duration_ms", ["INT", { 
            default: 200, 
            min: 50, 
            max: 2000,
            step: 50 
        }], app).widget;
        this.pulseDurationWidget.name = "Pulse Duration (ms)";
        
        // Add output (will be INT initially)
        this.addOutput("value", "INT");
        
        // Create button widget
        this.pressButton = this.addWidget("button", "press_button", null, () => {
            this.onButtonPress();
        });
        this.pressButton.label = "TRIGGER";
        this.pressButton.serialize = false;
        
        // Create display widget to show current value
        // NOTE: Name must be "value" for compatibility with NodeBypasser input reading
        this.displayWidget = ComfyWidgets["INT"](this, "value", ["INT", { 
            default: 0, 
            min: 0, 
            max: 999999,
            step: 1 
        }], app).widget;
        this.displayWidget.name = "value";
        
        // Make display widget read-only
        const originalCallback = this.displayWidget.callback;
        this.displayWidget.callback = () => {
            // Prevent manual changes - reset to actual value
            if (this.modeWidget.value === "Increment") {
                this.displayWidget.value = this._triggerValue;
            } else {
                // Store actual boolean for Pulse mode
                this.displayWidget.value = this._pulseState;
            }
        };
        
        this.isVirtualNode = true;
        this.serialize_widgets = true;
        this.color = "#2a363b";
        this.bgcolor = "#3e4a50";
        
        console.log("[MomentaryButton] Node created");
        
        // Call onConstructed after a brief delay to ensure full initialization
        setTimeout(() => {
            this.onConstructed();
        }, 10);
    }
    
    onConstructed() {
        this.__constructed__ = true;
        console.log("[MomentaryButton] Node fully constructed");
    }
    
    computeSize() {
        // Ensure size is always properly set
        if (!this.size || this.size.length !== 2) {
            this.size = [200, 130];
        }
        return this.size;
    }
    
    // Handle mode change
    onModeChange(newMode) {
        console.log(`[MomentaryButton] Mode changed to: ${newMode}`);
        
        // Update output type
        if (this.outputs && this.outputs[0]) {
            this.outputs[0].type = newMode === "Pulse" ? "BOOLEAN" : "INT";
            this.outputs[0].name = "value";
        }
        
        // Update display - use actual boolean for Pulse mode
        if (newMode === "Pulse") {
            this._pulseState = false;
            this.displayWidget.value = false;
        } else {
            this.displayWidget.value = this._triggerValue;
        }
        
        // Mark the graph as changed
        if (this.graph) {
            this.graph.change();
        }
    }
    
    // Handle button press
    onButtonPress() {
        const mode = this.modeWidget.value;
        
        if (mode === "Pulse") {
            // Clear any existing pulse timeout
            if (this._pulseTimeout) {
                clearTimeout(this._pulseTimeout);
            }
            
            // Send TRUE pulse
            this._pulseState = true;
            this.displayWidget.value = true;
            
            // Mark the graph as changed to trigger workflow updates
            if (this.graph) {
                this.graph.change();
            }
            
            // Visual feedback - flash green
            const originalBgColor = this.bgcolor;
            this.bgcolor = "#4a7a5a";
            this.setDirtyCanvas(true, true);
            
            // Auto-reset to FALSE after pulse duration
            const pulseDuration = this.pulseDurationWidget.value;
            this._pulseTimeout = setTimeout(() => {
                this._pulseState = false;
                this.displayWidget.value = false;
                
                // Mark the graph as changed again to trigger the FALSE transition
                if (this.graph) {
                    this.graph.change();
                }
                
                // Reset visual feedback
                this.bgcolor = originalBgColor;
                this.setDirtyCanvas(true, true);
                
                this._pulseTimeout = null;
            }, pulseDuration);
            
        } else {
            // Increment mode
            this._triggerValue = (this._triggerValue + 1) % 1000000;
            this.displayWidget.value = this._triggerValue;
            
            // Visual feedback - flash blue
            const originalBgColor = this.bgcolor;
            this.bgcolor = "#5a7a9f";
            this.setDirtyCanvas(true, true);
            
            setTimeout(() => {
                this.bgcolor = originalBgColor;
                this.setDirtyCanvas(true, true);
            }, 150);
            
            // Mark the graph as changed
            if (this.graph) {
                this.graph.change();
            }
        }
    }
    
    // Get the current output value based on mode
    getCurrentValue() {
        return this.modeWidget.value === "Pulse" ? this._pulseState : this._triggerValue;
    }
    
    // Clean up on removal
    onRemoved() {
        if (this._pulseTimeout) {
            clearTimeout(this._pulseTimeout);
            this._pulseTimeout = null;
        }
    }
    
    // Provide the output value when other nodes read from this node
    onExecute() {
        this.setOutputData(0, this.getCurrentValue());
    }
    
    // Override getOutputData to make this node's output readable
    getOutputData(slot) {
        return this.getCurrentValue();
    }
    
    // Called when the node is drawn (for updates)
    onDrawBackground(ctx) {
        // Update output data continuously
        if (this.outputs && this.outputs[0]) {
            this.setOutputData(0, this.getCurrentValue());
        }
    }
    
    // Save state
    serialize() {
        const data = super.serialize();
        if (data) {
            data.widget_values = {
                trigger_value: this._triggerValue,
                mode: this.modeWidget.value,
                pulse_duration: this.pulseDurationWidget.value
            };
        }
        return data;
    }
    
    // Restore state
    configure(data) {
        super.configure(data);  // Critical for copy-paste to work!
        
        // Handle both old format (direct properties) and new format (widget_values)
        const values = data.widget_values || data;
        
        if (values.trigger_value !== undefined) {
            this._triggerValue = values.trigger_value;
        }
        if (values.mode !== undefined) {
            this.modeWidget.value = values.mode;
            this.onModeChange(values.mode);
        }
        if (values.pulse_duration !== undefined) {
            this.pulseDurationWidget.value = values.pulse_duration;
        }
        
        // Update display based on current mode
        // Pulse mode always starts at false
        if (this.modeWidget.value === "Pulse") {
            this._pulseState = false;
            this.displayWidget.value = false;
        } else {
            this.displayWidget.value = this._triggerValue;
        }
    }
}

// Node metadata
MomentaryButton.title = "Momentary Button";
MomentaryButton.type = "NV/MomentaryButton";
MomentaryButton.category = "NV_Utils";
MomentaryButton.description = "Button that outputs INT (increment) or BOOLEAN (momentary pulse: true→false) - compatible with bypass/enable inputs";

// Register the extension
console.log("[MomentaryButton] Extension loading...");
app.registerExtension({
    name: "NV_Comfy_Utils.MomentaryButton",
    registerCustomNodes() {
        console.log("[MomentaryButton] Registering node type:", MomentaryButton.type);
        LiteGraph.registerNodeType(MomentaryButton.type, MomentaryButton);
        console.log("[MomentaryButton] Node type registered successfully");
    }
});
