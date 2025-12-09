"""
Slack Error Handler - Server Extension for ComfyUI
Sends Slack notification when workflow execution fails.

Automatically activates if these env vars are set:
- SLACK_BOT_TOKEN: Your Slack bot token
- SLACK_ERROR_CHANNEL: Channel or user ID for error notifications

If either is missing, this module does nothing (graceful degradation).

Supports loading from .env file in NV_Comfy_Utils directory (requires python-dotenv).
"""

import os
import time
from pathlib import Path

# Try to load .env file from NV_Comfy_Utils directory
try:
    from dotenv import load_dotenv
    # This file is at: NV_Comfy_Utils/src/KNF_Utils/slack_error_handler.py
    # .env file is at: NV_Comfy_Utils/.env
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"[NV_SlackErrorHandler] Loaded .env from {env_file}")
except ImportError:
    # python-dotenv not installed, rely on system env vars
    pass

# Check environment variables - if not set, skip everything
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_ERROR_CHANNEL = os.environ.get("SLACK_ERROR_CHANNEL", "")

# Only proceed if both credentials are configured
if SLACK_BOT_TOKEN and SLACK_ERROR_CHANNEL:

    # Try to import slack-sdk
    try:
        from slack_sdk import WebClient
        from slack_sdk.errors import SlackApiError
        SLACK_SDK_AVAILABLE = True
    except ImportError:
        SLACK_SDK_AVAILABLE = False
        print("[NV_SlackErrorHandler] slack-sdk not installed. Run: pip install slack-sdk")

    if SLACK_SDK_AVAILABLE:
        import execution

        # Store reference to original method
        _original_handle_execution_error = execution.PromptExecutor.handle_execution_error

        def _send_slack_error(error_details: dict):
            """Send error notification to Slack."""
            try:
                import socket
                computer_name = socket.gethostname()

                node_id = error_details.get("node_id", "unknown")
                node_type = error_details.get("node_type", "unknown")
                exception_message = error_details.get("exception_message", "No message")

                # Build simple error message with machine name
                message = (
                    f"{computer_name}\n"
                    f"ComfyUI Error\n"
                    f"Node: {node_type} (id: {node_id})\n"
                    f"Error: {exception_message}"
                )

                client = WebClient(token=SLACK_BOT_TOKEN)
                client.chat_postMessage(channel=SLACK_ERROR_CHANNEL, text=message)
                print(f"[NV_SlackErrorHandler] Sent error notification to {SLACK_ERROR_CHANNEL}")

            except SlackApiError as e:
                error_msg = e.response.get('error', str(e)) if hasattr(e, 'response') else str(e)
                print(f"[NV_SlackErrorHandler] Slack API error: {error_msg}")
            except Exception as e:
                print(f"[NV_SlackErrorHandler] Failed to send notification: {e}")

        def _wrapped_handle_execution_error(self, prompt_id, prompt, current_outputs, executed, error, ex):
            """Wrapped error handler that sends Slack notification before calling original."""

            # Build error details for Slack
            node_id = error.get("node_id", "unknown") if error else "unknown"
            node_type = prompt.get(node_id, {}).get("class_type", "unknown") if prompt and node_id != "unknown" else "unknown"
            exception_message = error.get("exception_message", str(ex)) if error else str(ex)

            error_details = {
                "node_id": node_id,
                "node_type": node_type,
                "exception_message": exception_message,
            }

            # Send Slack notification
            _send_slack_error(error_details)

            # Call original error handler
            return _original_handle_execution_error(self, prompt_id, prompt, current_outputs, executed, error, ex)

        # Apply the wrapper
        execution.PromptExecutor.handle_execution_error = _wrapped_handle_execution_error
        print(f"[NV_SlackErrorHandler] Enabled - errors will notify {SLACK_ERROR_CHANNEL}")

        # ===== SUCCESS NOTIFICATION SETUP =====

        # Track execution start times by prompt_id
        _execution_start_times = {}

        def _format_duration(seconds: float) -> str:
            """Format duration as human-readable string."""
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                minutes = int(seconds // 60)
                secs = int(seconds % 60)
                return f"{minutes}m {secs}s"
            else:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                return f"{hours}h {minutes}m {secs}s"

        def _send_slack_success(prompt_id: str, duration_seconds: float):
            """Send success notification to Slack."""
            try:
                import socket
                computer_name = socket.gethostname()
                formatted_duration = _format_duration(duration_seconds)

                message = (
                    f"{computer_name}\n"
                    f"Workflow Complete\n"
                    f"Prompt ID: {prompt_id}\n"
                    f"Execution Time: {formatted_duration}"
                )

                client = WebClient(token=SLACK_BOT_TOKEN)
                client.chat_postMessage(channel=SLACK_ERROR_CHANNEL, text=message)
                print(f"[NV_SlackErrorHandler] Sent success notification to {SLACK_ERROR_CHANNEL}")

            except SlackApiError as e:
                error_msg = e.response.get('error', str(e)) if hasattr(e, 'response') else str(e)
                print(f"[NV_SlackErrorHandler] Slack API error: {error_msg}")
            except Exception as e:
                print(f"[NV_SlackErrorHandler] Failed to send success notification: {e}")

        # Wrap add_message to intercept execution events
        _original_add_message = execution.PromptExecutor.add_message

        def _wrapped_add_message(self, event, data, broadcast=False):
            """Wrapped add_message that tracks execution timing."""
            # Track start time
            if event == "execution_start":
                prompt_id = data.get("prompt_id")
                if prompt_id:
                    _execution_start_times[prompt_id] = time.time()
            # Send success notification with timing
            elif event == "execution_success":
                prompt_id = data.get("prompt_id")
                if prompt_id:
                    start_time = _execution_start_times.pop(prompt_id, None)
                    if start_time:
                        duration = time.time() - start_time
                        _send_slack_success(prompt_id, duration)

            # Call original
            return _original_add_message(self, event, data, broadcast)

        execution.PromptExecutor.add_message = _wrapped_add_message
        print(f"[NV_SlackErrorHandler] Success notifications enabled")

        # Send startup notification
        def _send_startup_notification():
            """Send startup notification to Slack."""
            try:
                import socket
                computer_name = socket.gethostname()
                message = f"{computer_name} has started up ComfyUI"

                client = WebClient(token=SLACK_BOT_TOKEN)
                client.chat_postMessage(channel=SLACK_ERROR_CHANNEL, text=message)
                print(f"[NV_SlackErrorHandler] Sent startup notification to {SLACK_ERROR_CHANNEL}")

            except SlackApiError as e:
                error_msg = e.response.get('error', str(e)) if hasattr(e, 'response') else str(e)
                print(f"[NV_SlackErrorHandler] Slack API error on startup: {error_msg}")
            except Exception as e:
                print(f"[NV_SlackErrorHandler] Failed to send startup notification: {e}")

        _send_startup_notification()

else:
    # Silently skip if credentials not configured
    pass
