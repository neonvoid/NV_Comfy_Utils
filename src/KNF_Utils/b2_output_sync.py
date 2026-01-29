"""
NV B2 Output Sync

Pushes local folders TO a Backblaze B2 bucket using rclone via subprocess.

PUSH ONLY - This node only uploads local files to B2.
No pulling, no awareness of startup processes.
Single responsibility: sync local folder → B2 bucket.

Requirements:
- rclone installed and in PATH
- rclone configured with a [b2] remote (in ~/.config/rclone/rclone.conf)
- B2_BUCKET environment variable set
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


class NV_B2OutputSync:
    """
    Pushes local output folder TO B2 bucket using rclone.

    Designed to be chained after save nodes (image/video) to automatically
    push outputs to cloud storage when workflow completes.

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
                    "tooltip": "copy: Add new/changed files only (safe). sync: Mirror exact state (may delete remote files not in local)."
                }),
                "dry_run": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Preview changes without actually syncing. Check console output for what would be transferred."
                }),
            },
            "optional": {
                # Pass-through input for chaining after save nodes
                "passthrough": (ANY_TYPE, {
                    "tooltip": "Connect to output of save nodes to chain sync after saves complete."
                }),
                # Remote name (allows different rclone remotes)
                "remote_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "rclone remote name (e.g., 'b2', 'zs_b2'). Empty = 'b2'. Must be configured in rclone."
                }),
                # Local source path
                "local_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Local folder to sync. Empty = ComfyUI output folder."
                }),
                # B2 destination path (within bucket)
                "b2_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Remote path in bucket. Empty = comfy_outputs/<hostname>/. Use 'custom:path' for exact path."
                }),
                # Filters
                "include_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Include filter pattern (e.g., '*.mp4' or '*.png'). Empty = all files."
                }),
                "exclude_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Exclude filter pattern (e.g., '*.tmp' or 'preview_*'). Empty = exclude nothing."
                }),
            }
        }

    RETURN_TYPES = (ANY_TYPE, "STRING", "STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("passthrough", "status", "details", "files_transferred", "success")
    OUTPUT_NODE = True
    FUNCTION = "sync_to_b2"
    CATEGORY = "NV_Utils/Cloud"
    DESCRIPTION = "Pushes local output folder TO B2 bucket using rclone. Chain after save nodes."

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

    def _get_b2_bucket(self) -> Tuple[Optional[str], str]:
        """
        Get B2 bucket name from environment variable.

        Returns:
            Tuple of (bucket_name: Optional[str], message: str)
        """
        bucket = os.environ.get("B2_BUCKET")
        if not bucket:
            return None, "B2_BUCKET environment variable not set."
        return bucket, f"Using bucket: {bucket}"

    def _resolve_paths(
        self,
        local_path: str,
        b2_path: str,
        bucket: str,
        remote_name: str = "b2"
    ) -> Tuple[str, str]:
        """
        Resolve local and remote paths to full paths.

        Args:
            local_path: User-provided local path (empty = output folder)
            b2_path: User-provided B2 path (empty = default)
            bucket: B2 bucket name
            remote_name: rclone remote name (default: "b2")

        Returns:
            Tuple of (resolved_local_path, resolved_remote_path)
        """
        # Resolve local path
        if local_path and local_path.strip():
            local_path = local_path.strip()
            if os.path.isabs(local_path):
                resolved_local = local_path
            else:
                # Relative to ComfyUI output directory
                resolved_local = os.path.join(folder_paths.get_output_directory(), local_path)
        else:
            # Default: ComfyUI output folder
            resolved_local = folder_paths.get_output_directory()

        # Normalize path for cross-platform
        resolved_local = str(Path(resolved_local).resolve())

        # Resolve B2 path
        if b2_path and b2_path.strip():
            b2_path = b2_path.strip()
            if b2_path.startswith("custom:"):
                # User wants exact path
                remote_subpath = b2_path[7:]  # Remove "custom:" prefix
            else:
                # Append to default path structure
                remote_subpath = f"comfy_outputs/{self.hostname}/{b2_path}"
        else:
            # Default: comfy_outputs/<hostname>/
            remote_subpath = f"comfy_outputs/{self.hostname}"

        # Ensure forward slashes for B2 paths
        remote_subpath = remote_subpath.replace("\\", "/")

        # Build full remote path: remote_name:bucket-name/path
        resolved_remote = f"{remote_name}:{bucket}/{remote_subpath}"

        return resolved_local, resolved_remote

    def _build_rclone_command(
        self,
        sync_mode: str,
        local_path: str,
        remote_path: str,
        dry_run: bool,
        include_filter: str,
        exclude_filter: str
    ) -> list:
        """
        Build the rclone command with all options.

        Args:
            sync_mode: "copy" or "sync"
            local_path: Resolved local path
            remote_path: Resolved remote path
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

        # Add source and destination
        cmd.append(local_path)
        cmd.append(remote_path)

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

        if "connection" in stderr.lower() or "network" in stderr.lower():
            return f"{base_msg}. Network error. Check internet connection."

        if stderr.strip():
            # Truncate long error messages
            error_detail = stderr[:200] if len(stderr) > 200 else stderr
            return f"{base_msg}. Details: {error_detail}"

        return base_msg

    def sync_to_b2(
        self,
        sync_mode: str,
        dry_run: bool,
        passthrough=None,
        remote_name: str = "",
        local_path: str = "",
        b2_path: str = "",
        include_filter: str = "",
        exclude_filter: str = ""
    ) -> Tuple:
        """
        Execute the B2 sync operation.

        Returns:
            Tuple of (passthrough, status, details, files_transferred, success)
        """
        # Default remote_name to "b2" if empty
        if not remote_name or not remote_name.strip():
            remote_name = "b2"
        else:
            remote_name = remote_name.strip()

        print("\n" + "=" * 60)
        print(f"[NV_B2OutputSync] Starting B2 output sync (remote: {remote_name})...")
        print("=" * 60)

        # Check rclone availability (don't cache - remote may change)
        available, rclone_msg = self._check_rclone_available(remote_name)
        print(f"[NV_B2OutputSync] {rclone_msg}")

        if not available:
            error_msg = f"rclone not available: {rclone_msg}"
            print(f"[NV_B2OutputSync] ERROR: {error_msg}")
            return (passthrough, "ERROR", error_msg, 0, False)

        # Get B2 bucket
        bucket, bucket_msg = self._get_b2_bucket()
        if not bucket:
            print(f"[NV_B2OutputSync] ERROR: {bucket_msg}")
            return (passthrough, "ERROR", bucket_msg, 0, False)

        print(f"[NV_B2OutputSync] {bucket_msg}")

        # Resolve paths
        resolved_local, resolved_remote = self._resolve_paths(local_path, b2_path, bucket, remote_name)

        # Validate local path exists
        if not os.path.exists(resolved_local):
            error_msg = f"Local path does not exist: {resolved_local}"
            print(f"[NV_B2OutputSync] ERROR: {error_msg}")
            return (passthrough, "ERROR", error_msg, 0, False)

        print(f"[NV_B2OutputSync] Local: {resolved_local}")
        print(f"[NV_B2OutputSync] Remote: {resolved_remote}")
        print(f"[NV_B2OutputSync] Mode: {sync_mode}" + (" (DRY RUN)" if dry_run else ""))

        if include_filter:
            print(f"[NV_B2OutputSync] Include: {include_filter}")
        if exclude_filter:
            print(f"[NV_B2OutputSync] Exclude: {exclude_filter}")

        # Build command
        cmd = self._build_rclone_command(
            sync_mode=sync_mode,
            local_path=resolved_local,
            remote_path=resolved_remote,
            dry_run=dry_run,
            include_filter=include_filter,
            exclude_filter=exclude_filter
        )

        print(f"[NV_B2OutputSync] Command: {' '.join(cmd)}")
        print("-" * 60)

        # Execute rclone
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout for large syncs
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
                    details = f"Dry run complete. Would transfer {files_transferred} file(s) to {resolved_remote}"
                else:
                    status = "SUCCESS"
                    if files_transferred == 0:
                        details = f"Sync complete. No new files to transfer (folder may be empty or already synced)."
                    else:
                        details = f"Sync complete. Transferred {files_transferred} file(s) to {resolved_remote}"

                print(f"[NV_B2OutputSync] {status}: {details}")
                print("=" * 60 + "\n")
                return (passthrough, status, details, files_transferred, True)
            else:
                # Handle specific error codes
                error_details = self._interpret_error(result.returncode, stderr)
                print(f"[NV_B2OutputSync] FAILED: {error_details}")
                print("=" * 60 + "\n")
                return (passthrough, "FAILED", error_details, files_transferred, False)

        except subprocess.TimeoutExpired:
            error_msg = "Sync timed out after 30 minutes. Try syncing smaller batches with filters."
            print(f"[NV_B2OutputSync] TIMEOUT: {error_msg}")
            print("=" * 60 + "\n")
            return (passthrough, "TIMEOUT", error_msg, 0, False)

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"[NV_B2OutputSync] ERROR: {error_msg}")
            print("=" * 60 + "\n")
            return (passthrough, "ERROR", error_msg, 0, False)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_B2OutputSync": NV_B2OutputSync,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_B2OutputSync": "NV B2 Output Sync",
}
