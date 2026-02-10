"""
NV B2 Input Sync

Pulls files FROM a Backblaze B2 bucket to local folder using rclone via subprocess.

PULL ONLY - This node only downloads files from B2 to local.
Single responsibility: sync B2 bucket → local folder.

Use cases:
- Pull models/checkpoints to serverless GPU instances on startup
- Sync input assets before processing workflows
- Download project files from centralized B2 storage

Requirements:
- rclone installed and in PATH
- EITHER: rclone configured with a named remote (e.g. [b2]) + B2_BUCKET env var or bucket_name input
- OR: inline B2 credentials (b2_app_key_id + b2_app_key) + bucket_name — no pre-configured remote needed
"""

import os
import subprocess
import socket
import shutil
import re
from pathlib import Path
from typing import Tuple, Optional

import folder_paths

# Import IO.ANY for wildcard pass-through
try:
    from comfy.comfy_types.node_typing import IO
    ANY_TYPE = IO.ANY
except ImportError:
    # Fallback for older ComfyUI versions
    ANY_TYPE = "*"


class NV_B2InputSync:
    """
    Pulls files FROM B2 bucket to local folder using rclone.

    Designed to run at workflow start to download required assets
    (models, inputs, configs) from cloud storage before processing.

    Uses rclone with pre-configured [b2] remote.
    """

    def __init__(self):
        self._rclone_available = None  # Lazy check, cached
        self.hostname = socket.gethostname()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sync_mode": (["copy", "sync"], {
                    "default": "copy",
                    "tooltip": "copy: Add new/changed files only (safe). sync: Mirror exact state (may delete local files not in remote)."
                }),
                "dry_run": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Preview changes without actually downloading. Check console output for what would be transferred."
                }),
            },
            "optional": {
                # Pass-through input for chaining
                "passthrough": (ANY_TYPE, {
                    "tooltip": "Connect to chain this node with other operations."
                }),
                # Remote name (allows different rclone remotes)
                "remote_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "rclone remote name (e.g., 'b2', 'b2-inputs'). Empty = 'b2'. Must be configured in rclone. Ignored when app key credentials are provided."
                }),
                # Bucket name (overrides B2_BUCKET env var)
                "bucket_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "B2 bucket name. Overrides B2_BUCKET env var. Leave empty for bucket-scoped remotes (where bucket is implicit in the remote config)."
                }),
                # B2 app key ID for inline credential config
                "b2_app_key_id": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "B2 Application Key ID. If provided with app_key, creates a dynamic rclone remote — no pre-configured remote needed."
                }),
                # B2 app key for inline credential config
                "b2_app_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "B2 Application Key. Used with app_key_id to create a dynamic rclone remote."
                }),
                # B2 source path (within bucket)
                "b2_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Remote path in bucket to download from. Use 'custom:path' for exact path, otherwise uses comfy_inputs/<hostname>/."
                }),
                # Local destination path
                "local_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Local folder to download to. Empty = ComfyUI input folder."
                }),
                # Filters
                "include_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Include filter pattern (e.g., '*.safetensors' or '*.png'). Empty = all files."
                }),
                "exclude_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Exclude filter pattern (e.g., '*.tmp' or 'preview_*'). Empty = exclude nothing."
                }),
                # Option to create local folder if it doesn't exist
                "create_local_folder": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Create local destination folder if it doesn't exist."
                }),
            }
        }

    RETURN_TYPES = (ANY_TYPE, "STRING", "STRING", "STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("passthrough", "local_path", "status", "details", "files_transferred", "success")
    OUTPUT_NODE = True
    FUNCTION = "sync_from_b2"
    CATEGORY = "NV_Utils/Cloud"
    DESCRIPTION = "Pulls files FROM B2 bucket to local folder using rclone. Use at workflow start to fetch assets."

    def _check_rclone_available(self, remote_name: str = "b2") -> Tuple[bool, str]:
        """
        Check if rclone is installed and has the specified remote configured.

        Args:
            remote_name: Name of the rclone remote to check for (default: "b2")

        Returns:
            Tuple of (available: bool, message: str)
        """
        # Check if rclone binary exists
        rclone_path = shutil.which("rclone")
        if not rclone_path:
            return False, "rclone not found in PATH. Install rclone or add to PATH."

        # Check if the specified remote is configured
        try:
            result = subprocess.run(
                ["rclone", "listremotes"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return False, f"rclone error: {result.stderr}"

            remotes = result.stdout.strip().split('\n')
            remote_with_colon = f"{remote_name}:"
            if remote_with_colon not in remotes:
                return False, f"rclone [{remote_name}] remote not configured. Run 'rclone config' to set up. Available: {', '.join(remotes)}"

            return True, f"rclone available at {rclone_path}, using remote [{remote_name}]"

        except subprocess.TimeoutExpired:
            return False, "rclone command timed out"
        except Exception as e:
            return False, f"Error checking rclone: {str(e)}"

    def _get_b2_bucket(self, bucket_override: str = "") -> Tuple[Optional[str], str]:
        """
        Get B2 bucket name from input override or environment variable.

        Args:
            bucket_override: Bucket name from node input (takes priority over env var)

        Returns:
            Tuple of (bucket_name: Optional[str], message: str)
        """
        if bucket_override and bucket_override.strip():
            return bucket_override.strip(), f"Using bucket (from input): {bucket_override.strip()}"

        bucket = os.environ.get("B2_BUCKET")
        if bucket:
            return bucket, f"Using bucket (from env): {bucket}"

        # No bucket specified — valid for bucket-scoped remotes
        return None, "No bucket specified (using bucket-scoped remote or bucket implicit in remote config)."

    def _resolve_paths(
        self,
        b2_path: str,
        local_path: str,
        bucket: str,
        remote_name: str = "b2"
    ) -> Tuple[str, str]:
        """
        Resolve remote and local paths to full paths.

        Args:
            b2_path: User-provided B2 path (empty = default)
            local_path: User-provided local path (empty = input folder)
            bucket: B2 bucket name
            remote_name: rclone remote name (default: "b2")

        Returns:
            Tuple of (resolved_remote_path, resolved_local_path)
        """
        # Resolve B2 path (source)
        if b2_path and b2_path.strip():
            b2_path = b2_path.strip()
            if b2_path.startswith("custom:"):
                # User wants exact path
                remote_subpath = b2_path[7:]  # Remove "custom:" prefix
            else:
                # Append to default path structure
                remote_subpath = f"comfy_inputs/{self.hostname}/{b2_path}"
        else:
            # Default: comfy_inputs/<hostname>/
            remote_subpath = f"comfy_inputs/{self.hostname}"

        # Ensure forward slashes for B2 paths
        remote_subpath = remote_subpath.replace("\\", "/")

        # Build full remote path
        if bucket:
            resolved_remote = f"{remote_name}:{bucket}/{remote_subpath}"
        else:
            # Bucket-scoped remote — bucket is implicit in the remote config
            resolved_remote = f"{remote_name}:{remote_subpath}"

        # Resolve local path (destination)
        if local_path and local_path.strip():
            local_path = local_path.strip()
            if os.path.isabs(local_path):
                resolved_local = local_path
            else:
                # Relative to ComfyUI input directory
                resolved_local = os.path.join(folder_paths.get_input_directory(), local_path)
        else:
            # Default: ComfyUI input folder
            resolved_local = folder_paths.get_input_directory()

        # Normalize path for cross-platform
        resolved_local = str(Path(resolved_local).resolve())

        return resolved_remote, resolved_local

    def _build_rclone_command(
        self,
        sync_mode: str,
        remote_path: str,
        local_path: str,
        dry_run: bool,
        include_filter: str,
        exclude_filter: str
    ) -> list:
        """
        Build the rclone command with all options.

        Args:
            sync_mode: "copy" or "sync"
            remote_path: Resolved remote path (source)
            local_path: Resolved local path (destination)
            dry_run: Whether to do a dry run
            include_filter: Include pattern
            exclude_filter: Exclude pattern

        Returns:
            List of command arguments
        """
        cmd = ["rclone", sync_mode]

        # Add dry-run flag if requested
        if dry_run:
            cmd.append("--dry-run")

        # Add progress and stats for logging
        cmd.extend(["--progress", "--stats-one-line", "-v"])

        # Add include filter
        if include_filter and include_filter.strip():
            cmd.extend(["--include", include_filter.strip()])

        # Add exclude filter
        if exclude_filter and exclude_filter.strip():
            cmd.extend(["--exclude", exclude_filter.strip()])

        # Add source (remote) and destination (local)
        # NOTE: For download, remote comes first, local second
        cmd.append(remote_path)
        cmd.append(local_path)

        return cmd

    def _parse_transfer_count(self, output: str) -> int:
        """
        Parse rclone output to count transferred files.

        Args:
            output: Combined stdout/stderr from rclone

        Returns:
            Number of files transferred (estimate)
        """
        # Look for patterns like "Transferred:   5 / 5, 100%"
        match = re.search(r'Transferred:\s*(\d+)\s*/\s*\d+', output)
        if match:
            return int(match.group(1))

        # Alternative: count individual file transfer lines
        # rclone -v outputs lines like "file.mp4: Copied (new)"
        copied_count = len(re.findall(r': Copied \(', output))
        if copied_count > 0:
            return copied_count

        return 0

    def _interpret_error(self, return_code: int, stderr: str) -> str:
        """
        Interpret rclone error codes into human-readable messages.

        Args:
            return_code: Process return code
            stderr: Standard error output

        Returns:
            Human-readable error message
        """
        # Common rclone exit codes
        error_codes = {
            1: "Syntax or usage error",
            2: "Error not otherwise categorized",
            3: "Directory not found",
            4: "File not found",
            5: "Temporary error (retry may work)",
            6: "Less serious errors (some files not transferred)",
            7: "Fatal error",
            8: "Transfer limit reached",
            9: "No files to transfer",
        }

        base_msg = error_codes.get(return_code, f"Unknown error code {return_code}")

        # Add specific details from stderr
        if "InvalidSignature" in stderr or "Invalid key" in stderr:
            return f"{base_msg}. B2 credentials may be invalid."

        if "bucket not found" in stderr.lower():
            return f"{base_msg}. Bucket not found. Check B2_BUCKET environment variable."

        if "directory not found" in stderr.lower() or "not found" in stderr.lower():
            return f"{base_msg}. Remote path may not exist in bucket."

        if "connection" in stderr.lower() or "network" in stderr.lower():
            return f"{base_msg}. Network error. Check internet connection."

        if stderr.strip():
            # Truncate long error messages
            error_detail = stderr[:200] if len(stderr) > 200 else stderr
            return f"{base_msg}. Details: {error_detail}"

        return base_msg

    def sync_from_b2(
        self,
        sync_mode: str,
        dry_run: bool,
        passthrough=None,
        remote_name: str = "",
        bucket_name: str = "",
        b2_app_key_id: str = "",
        b2_app_key: str = "",
        b2_path: str = "",
        local_path: str = "",
        include_filter: str = "",
        exclude_filter: str = "",
        create_local_folder: bool = True
    ) -> Tuple:
        """
        Execute the B2 download operation.

        Returns:
            Tuple of (passthrough, local_path, status, details, files_transferred, success)
        """
        # Determine if using inline credentials (dynamic remote)
        use_dynamic = bool(b2_app_key_id and b2_app_key_id.strip() and b2_app_key and b2_app_key.strip())
        subprocess_env = None

        if use_dynamic:
            # Dynamic remote via rclone env var config — no pre-configured remote needed
            remote_name = "b2dyn"
            subprocess_env = os.environ.copy()
            subprocess_env["RCLONE_CONFIG_B2DYN_TYPE"] = "b2"
            subprocess_env["RCLONE_CONFIG_B2DYN_ACCOUNT"] = b2_app_key_id.strip()
            subprocess_env["RCLONE_CONFIG_B2DYN_KEY"] = b2_app_key.strip()

            print("\n" + "=" * 60)
            print(f"[NV_B2InputSync] Starting B2 input sync (dynamic remote with inline credentials)...")
            print("=" * 60)
        else:
            # Default remote_name to "b2" if empty
            if not remote_name or not remote_name.strip():
                remote_name = "b2"
            else:
                remote_name = remote_name.strip()

            print("\n" + "=" * 60)
            print(f"[NV_B2InputSync] Starting B2 input sync (download, remote: {remote_name})...")
            print("=" * 60)

            # Check rclone availability (don't cache - remote may change)
            available, rclone_msg = self._check_rclone_available(remote_name)
            print(f"[NV_B2InputSync] {rclone_msg}")

            if not available:
                error_msg = f"rclone not available: {rclone_msg}"
                print(f"[NV_B2InputSync] ERROR: {error_msg}")
                return (passthrough, "", "ERROR", error_msg, 0, False)

        # Check rclone binary exists (needed for both modes)
        if use_dynamic:
            rclone_path = shutil.which("rclone")
            if not rclone_path:
                error_msg = "rclone not found in PATH. Install rclone or add to PATH."
                print(f"[NV_B2InputSync] ERROR: {error_msg}")
                return (passthrough, "", "ERROR", error_msg, 0, False)

        # Get B2 bucket
        bucket, bucket_msg = self._get_b2_bucket(bucket_name)
        print(f"[NV_B2InputSync] {bucket_msg}")

        # Bucket is required when using dynamic credentials
        if use_dynamic and not bucket:
            error_msg = "bucket_name is required when using inline credentials (no bucket-scoped remote)."
            print(f"[NV_B2InputSync] ERROR: {error_msg}")
            return (passthrough, "", "ERROR", error_msg, 0, False)

        # Resolve paths
        resolved_remote, resolved_local = self._resolve_paths(b2_path, local_path, bucket, remote_name)

        # Create local folder if needed and requested
        if not os.path.exists(resolved_local):
            if create_local_folder:
                try:
                    os.makedirs(resolved_local, exist_ok=True)
                    print(f"[NV_B2InputSync] Created local folder: {resolved_local}")
                except Exception as e:
                    error_msg = f"Failed to create local folder {resolved_local}: {str(e)}"
                    print(f"[NV_B2InputSync] ERROR: {error_msg}")
                    return (passthrough, "", "ERROR", error_msg, 0, False)
            else:
                error_msg = f"Local path does not exist: {resolved_local}"
                print(f"[NV_B2InputSync] ERROR: {error_msg}")
                return (passthrough, "", "ERROR", error_msg, 0, False)

        print(f"[NV_B2InputSync] Remote: {resolved_remote}")
        print(f"[NV_B2InputSync] Local: {resolved_local}")
        print(f"[NV_B2InputSync] Mode: {sync_mode}" + (" (DRY RUN)" if dry_run else ""))

        if include_filter:
            print(f"[NV_B2InputSync] Include: {include_filter}")
        if exclude_filter:
            print(f"[NV_B2InputSync] Exclude: {exclude_filter}")

        # Build command
        cmd = self._build_rclone_command(
            sync_mode=sync_mode,
            remote_path=resolved_remote,
            local_path=resolved_local,
            dry_run=dry_run,
            include_filter=include_filter,
            exclude_filter=exclude_filter
        )

        print(f"[NV_B2InputSync] Command: {' '.join(cmd)}")
        print("-" * 60)

        # Execute rclone
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout for large downloads
                env=subprocess_env  # None uses inherited env, or custom env for dynamic remotes
            )

            # Parse output for transfer stats
            stdout = result.stdout
            stderr = result.stderr

            # Log output
            if stdout:
                for line in stdout.split('\n'):
                    if line.strip():
                        print(f"[rclone] {line}")

            if stderr:
                for line in stderr.split('\n'):
                    if line.strip():
                        print(f"[rclone stderr] {line}")

            # Count transferred files from output
            files_transferred = self._parse_transfer_count(stdout + stderr)

            if result.returncode == 0:
                if dry_run:
                    status = "DRY_RUN_OK"
                    details = f"Dry run complete. Would download {files_transferred} file(s) from {resolved_remote}"
                else:
                    status = "SUCCESS"
                    if files_transferred == 0:
                        details = f"Sync complete. No new files to download (folder may be empty or already synced)."
                    else:
                        details = f"Sync complete. Downloaded {files_transferred} file(s) to {resolved_local}"

                print(f"[NV_B2InputSync] {status}: {details}")
                print("=" * 60 + "\n")
                return (passthrough, resolved_local, status, details, files_transferred, True)
            else:
                # Handle specific error codes
                error_details = self._interpret_error(result.returncode, stderr)
                print(f"[NV_B2InputSync] FAILED: {error_details}")
                print("=" * 60 + "\n")
                return (passthrough, resolved_local, "FAILED", error_details, files_transferred, False)

        except subprocess.TimeoutExpired:
            error_msg = "Download timed out after 30 minutes. Try downloading smaller batches with filters."
            print(f"[NV_B2InputSync] TIMEOUT: {error_msg}")
            print("=" * 60 + "\n")
            return (passthrough, resolved_local, "TIMEOUT", error_msg, 0, False)

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"[NV_B2InputSync] ERROR: {error_msg}")
            print("=" * 60 + "\n")
            return (passthrough, resolved_local, "ERROR", error_msg, 0, False)


# Node registration (will be imported by nodes.py)
NODE_CLASS_MAPPINGS = {
    "NV_B2InputSync": NV_B2InputSync,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_B2InputSync": "NV B2 Input Sync",
}
