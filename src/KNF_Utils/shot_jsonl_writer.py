"""Atomic JSONL append for the agent inbox.

Multi-machine portability: each render writes one line to
`{inbox_dir}/{MACHINE_ID}.jsonl`. Records never overwrite, never reorder.
Local consolidator script (Phase 1.5) drains the inbox into the agent's
prompt-cached canon.
"""

import json
import os
import socket
import time


def get_machine_id():
    """Stable per-machine identifier — hostname is good enough for our scale.

    Override via env var NV_MACHINE_ID if hostname collides across machines
    (e.g. cloud-GPU pools recycle generic hostnames).
    """
    override = os.environ.get("NV_MACHINE_ID", "").strip()
    if override:
        return override
    return socket.gethostname() or "unknown"


class _InboxLock:
    """Same O_EXCL pattern as sweep_recorder. Per-file lock so two workflow
    nodes appending to the same machine inbox don't interleave bytes.
    """
    def __init__(self, path, timeout=10.0, poll=0.05):
        self.lock_path = path + ".lock"
        self.timeout = timeout
        self.poll = poll
        self.fd = None

    def __enter__(self):
        deadline = time.time() + self.timeout
        while True:
            try:
                self.fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.time() > deadline:
                    raise TimeoutError(f"inbox lock timeout: {self.lock_path}")
                time.sleep(self.poll)

    def __exit__(self, *args):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            try:
                os.unlink(self.lock_path)
            except OSError:
                pass


def resolve_inbox_path(inbox_dir):
    """Substitute {MACHINE_ID} placeholder if present; else append filename.
    Ensures parent dir exists.
    """
    machine_id = get_machine_id()
    if "{MACHINE_ID}" in inbox_dir:
        path = inbox_dir.replace("{MACHINE_ID}", machine_id)
    elif inbox_dir.endswith(".jsonl"):
        path = inbox_dir
    else:
        path = os.path.join(inbox_dir, f"{machine_id}.jsonl")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def append_record(inbox_path, record):
    """Append one JSON line. Lock-protected. Never raises on disk errors —
    prints a warning so the workflow doesn't abort over a missing logs dir.
    """
    line = json.dumps(record, ensure_ascii=False) + "\n"
    try:
        with _InboxLock(inbox_path):
            with open(inbox_path, "a", encoding="utf-8") as f:
                f.write(line)
        return True
    except Exception as e:
        print(f"[shot_jsonl_writer] Warning: failed to append record to {inbox_path}: {e}")
        return False


def count_records(inbox_path):
    """Cheap line count for record_count output. Returns 0 if file missing."""
    if not os.path.exists(inbox_path):
        return 0
    try:
        with open(inbox_path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0
