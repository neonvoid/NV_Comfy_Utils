import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Add stable naming functionality to existing nodes
app.registerExtension({
    name: "NV_Comfy_Utils.StableNaming",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Add stable naming to Get/Set nodes
        if (nodeData.name === "GetVariableNode" || nodeData.name === "SetVariableNode") {
            const originalOnNodeCreated = nodeData.onNodeCreated;
            
            nodeData.onNodeCreated = function(node) {
                // Call original function if it exists
                if (originalOnNodeCreated) {
                    originalOnNodeCreated.call(this, node);
                }
                
                // Ensure properties exist
                if (!node.properties) {
                    node.properties = {};
                }
                
                // Add custom name widget
                const customNameWidget = ComfyWidgets["STRING"](node, "custom_name", ["STRING", { 
                    default: node.properties.custom_name || "",
                    placeholder: "Enter stable name for bypassing"
                }], app).widget;
                customNameWidget.name = "Custom Name";
                customNameWidget.computeSize = () => [200, 30];
                
                // Store reference to the custom name widget
                node._customNameWidget = customNameWidget;
                
                // Initialize stable custom name
                if (!node._stableCustomName) {
                    node._stableCustomName = node.properties.custom_name || "";
                }
                
                // Create a more robust widget change handler
                const originalOnWidgetChange = node.onWidgetChange;
                node.onWidgetChange = function(widget, value) {
                    // Handle custom name widget first
                    if (widget === customNameWidget) {
                        if (!this.properties) {
                            this.properties = {};
                        }
                        this.properties.custom_name = value;
                        console.log("[StableNaming] Custom name set to:", value);
                        return; // Don't call original handler for custom name
                    }
                    
                    // Call original handler for other widgets
                    if (originalOnWidgetChange) {
                        originalOnWidgetChange.call(this, widget, value);
                    }
                };
                
                // Override the node's configure method to preserve custom name
                const originalConfigure = node.configure;
                node.configure = function(info) {
                    // Save custom name before configuration
                    const customName = this.properties ? this.properties.custom_name : "";
                    
                    // Call original configure
                    if (originalConfigure) {
                        originalConfigure.call(this, info);
                    }
                    
                    // Restore custom name after configuration
                    if (customName && this.properties) {
                        this.properties.custom_name = customName;
                    }
                    
                    // Restore custom name widget value
                    if (this._customNameWidget) {
                        this._customNameWidget.value = customName || "";
                    }
                };
                
                // Override any method that might reset the node
                const originalOnResize = node.onResize;
                node.onResize = function(size) {
                    if (originalOnResize) {
                        originalOnResize.call(this, size);
                    }
                    // Ensure custom name is preserved
                    this._preserveCustomName();
                };
                
                // Add a method to preserve custom name
                node._preserveCustomName = function() {
                    if (this.properties && this.properties.custom_name && this._customNameWidget) {
                        this._customNameWidget.value = this.properties.custom_name;
                    }
                };
                
                // Monitor for property changes and restore custom name
                const originalSetProperty = node.setProperty;
                if (originalSetProperty) {
                    node.setProperty = function(name, value) {
                        const result = originalSetProperty.call(this, name, value);
                        // Restore custom name after any property change
                        setTimeout(() => this._preserveCustomName(), 0);
                        return result;
                    };
                }
                
                // Instead of fighting title changes, let's make the search more robust
                // We'll store the custom name in multiple places for reliability
                node._stableCustomName = null;
                
                // Override the widget change to store in multiple places
                const originalWidgetChange = node.onWidgetChange;
                node.onWidgetChange = function(widget, value) {
                    if (widget === customNameWidget) {
                        if (!this.properties) {
                            this.properties = {};
                        }
                        this.properties.custom_name = value;
                        this._stableCustomName = value; // Store in multiple places
                        console.log("[StableNaming] Custom name set to:", value);
                        return;
                    }
                    
                    if (originalWidgetChange) {
                        originalWidgetChange.call(this, widget, value);
                    }
                };
                
                // Add a periodic check to ensure custom name is preserved
                node._customNameCheckInterval = setInterval(() => {
                    if (this.properties && this.properties.custom_name && this._customNameWidget) {
                        if (this._customNameWidget.value !== this.properties.custom_name) {
                            console.log("[StableNaming] Restoring custom name:", this.properties.custom_name);
                            this._customNameWidget.value = this.properties.custom_name;
                        }
                    }
                    
                    // Also check if title has changed and restore it
                    if (node.title !== originalTitle) {
                        console.log("[StableNaming] Restoring original title:", originalTitle);
                        node.title = originalTitle;
                    }
                }, 500); // Check every 500ms for more responsiveness
                
                // Clean up interval when node is removed
                const originalOnRemoved = node.onRemoved;
                node.onRemoved = function() {
                    if (this._customNameCheckInterval) {
                        clearInterval(this._customNameCheckInterval);
                    }
                    if (originalOnRemoved) {
                        originalOnRemoved.call(this);
                    }
                };
                
                // Override serialize to include custom name
                const originalSerialize = node.serialize;
                node.serialize = function() {
                    const data = originalSerialize ? originalSerialize.call(this) : {};
                    if (this.properties && this.properties.custom_name) {
                        data.custom_name = this.properties.custom_name;
                    }
                    return data;
                };
                
                // Override configureFromData to restore custom name
                const originalConfigureFromData = node.configureFromData;
                node.configureFromData = function(data) {
                    if (originalConfigureFromData) {
                        originalConfigureFromData.call(this, data);
                    }
                    
                    // Restore custom name from saved data
                    if (data.custom_name) {
                        if (!this.properties) {
                            this.properties = {};
                        }
                        this.properties.custom_name = data.custom_name;
                        
                        // Update widget value
                        if (this._customNameWidget) {
                            this._customNameWidget.value = data.custom_name;
                        }
                    }
                };
            };
        }
    },
    
    loadedGraphNode(node) {
        // Ensure properties exist for all nodes
        if (!node.properties) {
            node.properties = {};
        }
        
        // Restore custom name functionality for existing nodes
        if ((node.type === "GetVariable" || node.type === "SetVariable") && !node._customNameWidget) {
            // Re-add the custom name widget if it's missing
            const customNameWidget = ComfyWidgets["STRING"](node, "custom_name", ["STRING", { 
                default: node.properties.custom_name || "",
                placeholder: "Enter stable name for bypassing"
            }], app).widget;
            customNameWidget.name = "Custom Name";
            customNameWidget.computeSize = () => [200, 30];
            node._customNameWidget = customNameWidget;
            
            // Add the widget change handler
            const originalOnWidgetChange = node.onWidgetChange;
            node.onWidgetChange = function(widget, value) {
                if (widget === customNameWidget) {
                    if (!this.properties) {
                        this.properties = {};
                    }
                    this.properties.custom_name = value;
                    console.log("[StableNaming] Custom name restored to:", value);
                    return;
                }
                
                if (originalOnWidgetChange) {
                    originalOnWidgetChange.call(this, widget, value);
                }
            };
        }
    }
});

console.log("[NV_Comfy_Utils] Stable naming extension loaded");
