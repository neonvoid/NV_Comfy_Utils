"""
Slack Notifier Node for ComfyUI
Sends a simple Slack message when executed with output file location.
"""

import os

# Conditional import of slack-sdk
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_SDK_AVAILABLE = True
except ImportError:
    SLACK_SDK_AVAILABLE = False
    WebClient = None
    SlackApiError = Exception

from comfy.comfy_types.node_typing import IO


class NV_SlackNotifier:
    """
    Sends a Slack notification when the node executes.
    Connect the trigger input to any node output to fire after that node completes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": (IO.ANY, {"tooltip": "Connect to any output to trigger notification after that node"}),
                "output_path": ("STRING", {"default": "", "tooltip": "Directory where output is saved"}),
                "filename": ("STRING", {"default": "", "tooltip": "Name of the output file"}),
                "channel": ("STRING", {"default": "#comfyui", "tooltip": "Slack channel (e.g., #channel or channel ID)"}),
            },
            "optional": {
                "slack_token": ("STRING", {"default": "", "tooltip": "Bot token (leave empty to use SLACK_BOT_TOKEN env var)"}),
                "message_prefix": ("STRING", {"default": "", "tooltip": "Optional label prefix for the message"}),
            }
        }

    RETURN_TYPES = (IO.ANY,)
    RETURN_NAMES = ("trigger_passthrough",)
    FUNCTION = "notify"
    CATEGORY = "NV_Utils/Notifications"
    OUTPUT_NODE = True
    DESCRIPTION = "Sends a Slack notification with output file path when executed"

    def notify(self, trigger, output_path, filename, channel,
               slack_token="", message_prefix=""):
        """Send Slack notification and pass through trigger."""

        if not SLACK_SDK_AVAILABLE:
            print("[NV_SlackNotifier] slack-sdk not installed. Run: pip install slack-sdk")
            return (trigger,)

        # Get token from input or environment variable
        token = slack_token.strip() if slack_token else os.environ.get("SLACK_BOT_TOKEN", "")

        if not token:
            print("[NV_SlackNotifier] No Slack token provided. Set SLACK_BOT_TOKEN env var or provide token in node.")
            return (trigger,)

        # Build full path
        if output_path and filename:
            full_path = os.path.join(output_path, filename)
        elif filename:
            full_path = filename
        elif output_path:
            full_path = output_path
        else:
            full_path = "(no path provided)"

        # Build message
        prefix = f"[{message_prefix}] " if message_prefix.strip() else ""
        message = f"{prefix}ComfyUI Complete\nOutput: {full_path}"

        # Send to Slack
        try:
            client = WebClient(token=token)
            response = client.chat_postMessage(channel=channel, text=message)
            print(f"[NV_SlackNotifier] Sent notification to {channel}")
        except SlackApiError as e:
            error_msg = e.response.get('error', str(e)) if hasattr(e, 'response') else str(e)
            print(f"[NV_SlackNotifier] Slack API error: {error_msg}")
        except Exception as e:
            print(f"[NV_SlackNotifier] Error sending notification: {e}")

        return (trigger,)
