import { app } from "../../scripts/app.js";
app.registerExtension({
    name: "NV_Comfy_Utils.DownloadVideo",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "CustomVideoSaver") return;

        // Capture video results from execution
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            if (message?.videos?.length) {
                this._lastVideo = message.videos[0];
            }
        };

        // Add right-click menu options
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            getExtraMenuOptions?.apply(this, arguments);

            const video = this._lastVideo;
            if (!video) return;

            const params = new URLSearchParams({
                filename: video.filename,
                subfolder: video.subfolder || "",
                type: video.type || "output",
            });
            const url = "/view?" + params;

            options.unshift(
                {
                    content: "Download Video",
                    callback: () => {
                        const a = document.createElement("a");
                        a.href = url;
                        a.setAttribute("download", video.filename);
                        document.body.append(a);
                        a.click();
                        requestAnimationFrame(() => a.remove());
                    },
                },
                {
                    content: "Open Video in New Tab",
                    callback: () => {
                        window.open(url, "_blank");
                    },
                },
                {
                    content: "Copy Output Path",
                    callback: async () => {
                        const path = video.fullpath || video.filename;
                        await navigator.clipboard.writeText(path);
                    },
                },
                null // separator
            );
        };
    },
});
