import { app } from "../../scripts/app.js";

console.log("[NV_Comfy_Utils] Loading simple variables...");

app.registerExtension({
    name: "NV_Comfy_Utils.SimpleVariables",
    
    registerCustomNodes() {
        console.log("[NV_Comfy_Utils] Registering simple variable nodes...");
        
        // Minimal Set Variable Node
        class SetVariableNode extends LGraphNode {
            constructor(title) {
                super(title);
                
                // Basic properties
                this.defaultVisibility = true;
                this.serialize_widgets = true;
                this.drawConnection = false;
                this.slotColor = "#FFF";
                this.canvas = app.canvas;
                
                // Properties object
                if (!this.properties) {
                    this.properties = {};
                }
                this.properties.previousName = "";
                
                // Initialize stable custom name
                this._stableCustomName = "";
                
                const node = this;

                // Simple text widget for variable name
                this.addWidget(
                    "text", 
                    "Variable Name", 
                    "", 
                    (value) => {
                        console.log("[SetVariableNode] Name changed to:", value);
                        // Don't change the title - keep it stable for NodeBypasser
                        this.properties.previousName = value;
                        this.updateGetters();
                    }
                );

                // Add stable naming widget for NodeBypasser
                this.addWidget(
                    "text", 
                    "Custom Name", 
                    "", 
                    (value) => {
                        console.log("[SetVariableNode] Custom name changed to:", value);
                        this._stableCustomName = value;
                        this.properties.custom_name = value;
                        // Update the title based on custom name instead of variable name
                        if (value !== '') {
                            this.title = "Set_" + value;
                        } else {
                            this.title = "Set Variable";
                        }
                    }
                );
                
                // Input and output
                this.addInput("*", "*");
                this.addOutput("*", '*');

                // Connection change handler
                this.onConnectionsChange = function(slotType, slot, isChangeConnect, link_info, output) {
                    if (slotType === 0) { // Input connection changed
                        this.updateGetters();
                    }
                };

                // Update all getter nodes that use this variable name
                this.updateGetters = function() {
                    if (!node.graph) return;
                    
                    const getters = node.graph._nodes.filter(otherNode => 
                        otherNode.type === 'GetVariableNode' && 
                        otherNode.widgets[0].value === this.widgets[0].value &&
                        this.widgets[0].value !== ''
                    );
                    
                    getters.forEach(getter => {
                        if (getter.updateType) {
                            getter.updateType();
                        }
                    });
                    
                    console.log("[SetVariableNode] Updated", getters.length, "getter nodes");
                };

                // This is a virtual node - frontend handles everything
                this.isVirtualNode = true;
                
                console.log("[SetVariableNode] Created");
            }
        }

        // Minimal Get Variable Node
        class GetVariableNode extends LGraphNode {
            constructor(title) {
                super(title);
                
                // Basic properties
                this.defaultVisibility = true;
                this.serialize_widgets = true;
                this.drawConnection = false;
                this.slotColor = "#FFF";
                this.canvas = app.canvas;
                
                // Properties object
                if (!this.properties) {
                    this.properties = {};
                }
                
                // Initialize stable custom name
                this._stableCustomName = "";
                
                const node = this;

                // Simple combo widget for variable name
                this.addWidget(
                    "combo",
                    "Variable Name",
                    "",
                    (value) => {
                        console.log("[GetVariableNode] Name changed to:", value);
                        // Don't change the title - keep it stable for NodeBypasser
                        this.updateType();
                    },
                    {
                        values: () => {
                            if (!node.graph) return [];
                            const setterNodes = node.graph._nodes.filter((otherNode) => otherNode.type == 'SetVariableNode');
                            return setterNodes.map((otherNode) => otherNode.widgets[0].value).filter(name => name !== '');
                        }
                    }
                );

                // Add stable naming widget for NodeBypasser
                this.addWidget(
                    "text", 
                    "Custom Name", 
                    "", 
                    (value) => {
                        console.log("[GetVariableNode] Custom name changed to:", value);
                        this._stableCustomName = value;
                        this.properties.custom_name = value;
                        // Update the title based on custom name instead of variable name
                        if (value !== '') {
                            this.title = "Get_" + value;
                        } else {
                            this.title = "Get Variable";
                        }
                    }
                );

                // Output
                this.addOutput("*", '*');

                // Find the corresponding setter node
                this.findSetter = function() {
                    const name = this.widgets[0].value;
                    if (!name || !node.graph) return null;
                    
                    const setter = node.graph._nodes.find(otherNode => 
                        otherNode.type === 'SetVariableNode' && 
                        otherNode.widgets[0].value === name
                    );
                    return setter;
                };

                // Update output type based on setter
                this.updateType = function() {
                    const setter = this.findSetter();
                    if (setter && setter.inputs[0].type) {
                        this.outputs[0].type = setter.inputs[0].type;
                        this.outputs[0].name = setter.inputs[0].type;
                        console.log("[GetVariableNode] Type updated to:", setter.inputs[0].type);
                    } else {
                        this.outputs[0].type = "*";
                        this.outputs[0].name = "*";
                    }
                };

                // Override the getInputLink method to pass through the setter's input link
                this.getInputLink = function(slot) {
                    const setter = this.findSetter();
                    if (setter && setter.inputs[slot]) {
                        const slotInfo = setter.inputs[slot];
                        const link = node.graph.links[slotInfo.link];
                        return link;
                    } else {
                        const errorMessage = "No SetVariableNode found for " + this.widgets[0].value + "(" + this.type + ")";
                        console.warn(errorMessage);
                    }
                    return null;
                };


                // This is a virtual node - frontend handles everything
                this.isVirtualNode = true;
                
                // onAdded method like KJ nodes
                this.onAdded = function(graph) {
                    // Called when node is added to graph
                };
                
                console.log("[GetVariableNode] Created");
            }
        }

        // Register the nodes
        LiteGraph.registerNodeType("SetVariableNode", SetVariableNode);
        LiteGraph.registerNodeType("GetVariableNode", GetVariableNode);
        
        console.log("[NV_Comfy_Utils] Simple variable nodes registered successfully");
    }
});
