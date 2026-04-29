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


_STALE_LOCK_AGE_SEC = 60.0   # ComfyUI crashes mid-write would otherwise wedge
                              # the inbox forever. 60s is far longer than any
                              # legitimate atomic append.


class _InboxLock:
    """Per-file lock via O_EXCL sentinel. Two workflow nodes appending to the
    same machine inbox can't interleave bytes.

    Stale-lock recovery: if an existing lock file is older than
    _STALE_LOCK_AGE_SEC, we assume the prior holder crashed and reclaim it.
    Single-user / sequential workflow makes this safe — concurrent writers
    on the SAME machine within 60s of each other don't happen in practice,
    and machine-isolated inbox files prevent cross-machine collisions.
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
                # Reclaim stale lock from a crashed prior holder.
                try:
                    age = time.time() - os.path.getmtime(self.lock_path)
                    if age > _STALE_LOCK_AGE_SEC:
                        os.unlink(self.lock_path)
                        continue  # retry the open immediately
                except OSError:
                    pass  # race with concurrent unlinker; let the retry loop handle it
                if time.time() > deadline:
                    raise TimeoutError(
                        f"inbox lock timeout: {self.lock_path} (held > {self.timeout}s, "
                        f"not stale-eligible). Delete the .lock manually if no writer is active."
                    )
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
    """Append one JSON line. Lock-protected. Atomic flush+fsync before lock
    release.

    RAISES on serialization or persistence failure — silent telemetry loss
    is exactly what poisons the agent corpus. Caller (NV_ShotRecord) is
    responsible for surfacing the error to the operator.

    `allow_nan=False` rejects non-finite floats (NaN/Infinity) — strict
    JSONL readers reject these, and they'd silently break the agent's
    aggregation pipeline.
    """
    # Fail fast on serialization issues so the operator sees the bug at
    # write time rather than discovering it weeks later in the agent corpus.
    line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
    with _InboxLock(inbox_path):
        with open(inbox_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # fsync can fail on some filesystems (network mounts) — flush
                # is the load-bearing call; fsync is best-effort durability.
                pass
    return True


def count_records(inbox_path):
    """Cheap line count for record_count output. Returns 0 if file missing."""
    if not os.path.exists(inbox_path):
        return 0
    try:
        with open(inbox_path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0
