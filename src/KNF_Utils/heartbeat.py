"""
Heartbeat Node for ComfyUI - Prevents serverless timeout by sending periodic websocket messages.
"""

import threading
import time
from comfy.comfy_types.node_typing import IO
import server


class NV_Heartbeat:
    """
    Sends periodic heartbeat messages to keep serverless connections alive.
    Place at start of workflow - heartbeat runs until workflow completes.
    """

    # Class-level tracking to prevent duplicate heartbeat threads
    _active_heartbeats = {}
    _lock = threading.Lock()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": (IO.ANY, {"tooltip": "Connect to any input to start heartbeat at that point"}),
                "interval_seconds": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 55,
                    "tooltip": "Seconds between heartbeat messages (keep under 60 for serverless)"
                }),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Enable/disable heartbeat"}),
            }
        }

    RETURN_TYPES = (IO.ANY,)
    RETURN_NAMES = ("trigger_passthrough",)
    FUNCTION = "start_heartbeat"
    CATEGORY = "NV_Utils/Serverless"
    DESCRIPTION = "Sends periodic heartbeat messages to prevent serverless deployment timeouts"

    def start_heartbeat(self, trigger, interval_seconds, enabled=True):
        if not enabled:
            return (trigger,)

        prompt_server = server.PromptServer.instance
        if not prompt_server:
            print("[NV_Heartbeat] Warning: PromptServer not available")
            return (trigger,)

        # Get current prompt_id from server
        prompt_id = getattr(prompt_server, 'last_prompt_id', None)
        thread_key = f"heartbeat_{id(threading.current_thread())}"

        # Stop any existing heartbeat for this execution
        self._stop_heartbeat(thread_key)

        # Create stop event and start heartbeat thread
        stop_event = threading.Event()

        def heartbeat_loop():
            while not stop_event.wait(timeout=interval_seconds):
                try:
                    prompt_server.send_sync("heartbeat", {
                        "timestamp": int(time.time() * 1000),
                        "prompt_id": prompt_id,
                        "message": "workflow_alive"
                    }, prompt_server.client_id)
                except Exception as e:
                    print(f"[NV_Heartbeat] Error sending heartbeat: {e}")
                    break

        thread = threading.Thread(target=heartbeat_loop, daemon=True, name="NV_Heartbeat")
        thread.start()

        with self._lock:
            self._active_heartbeats[thread_key] = (thread, stop_event)

        print(f"[NV_Heartbeat] Started heartbeat every {interval_seconds}s")
        return (trigger,)

    @classmethod
    def _stop_heartbeat(cls, key):
        with cls._lock:
            if key in cls._active_heartbeats:
                _, stop_event = cls._active_heartbeats.pop(key)
                stop_event.set()
