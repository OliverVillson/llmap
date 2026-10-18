#!/usr/bin/env python3
"""
Where qwen.sh keeps the big things (the weights, the vLLM venv and caches): the data folder.

  disk.py pick                  the folder to use: <mount>/qwen-data on the local filesystem with the
                                most free space that this user can write to ($HOME/qwen-data when that
                                is the home filesystem)
  disk.py free DIR              free GB on DIR's filesystem
  disk.py bigger DIR            local filesystems with more free space than DIR's (the folder to use, or
                                that this user cannot write there), and disks with no filesystem mounted
                                (lsblk), one line each
  disk.py model-gb MODEL HF_HOME [PYTHON]
                                GB still to download for MODEL: its size on the Hub (asked with PYTHON's
                                huggingface_hub; 31 if that fails) minus what is already in HF_HOME
"""
import json
import os
import shutil
import subprocess
import sys

LOCAL = {"ext4", "ext3", "xfs", "btrfs", "zfs", "f2fs"}
SKIP = ("/boot", "/snap", "/opt/claude-code", "/opt/env-runner")
GB = 1e9


def mounts():
    """(mount point, free bytes) for local disk filesystems."""
    out = []
    try:
        lines = open("/proc/mounts").read().splitlines()
    except OSError:
        return out
    seen = set()
    for line in lines:
        dev, target, fstype = line.split()[:3]
        target = target.replace("\\040", " ")
        if fstype not in LOCAL or target.startswith(SKIP) or dev in seen:
            continue
        seen.add(dev)
        try:
            out.append((target, shutil.disk_usage(target).free))
        except OSError:
            pass
    return out


def home_mount():
    home = os.path.realpath(os.path.expanduser("~"))
    best = "/"
    for target, _ in mounts():
        if (home == target or home.startswith(target.rstrip("/") + "/")) and len(target) > len(best):
            best = target
    return best


def candidate(target):
    """The data folder this user would use on that filesystem, or None if they cannot write there."""
    if target == home_mount():
        return os.path.join(os.path.expanduser("~"), "qwen-data")
    for d in (os.path.join(target, "qwen-data"), os.path.join(target, os.environ.get("USER", "")) + "/qwen-data"):
        parent = os.path.dirname(d.rstrip("/"))
        if os.path.isdir(d) and os.access(d, os.W_OK) or os.path.isdir(parent) and os.access(parent, os.W_OK):
            return d
    return None


def pick():
    best = None
    for target, free in mounts():
        d = candidate(target)
        if d and (best is None or free > best[1]):
            best = (d, free)
    return best[0] if best else os.path.join(os.path.expanduser("~"), "qwen-data")


def free_gb(d):
    while not os.path.exists(d):
        d = os.path.dirname(d) or "/"
    return shutil.disk_usage(d).free / GB


def bigger(d):
    mine = free_gb(d)
    for target, free in mounts():
        if free / GB > mine + 1:
            d = candidate(target)
            print(f"{target}: {free / GB:.0f} GB free, not writable by {os.environ.get('USER', 'you')}" if d is None else f"{target}: {free / GB:.0f} GB free; use it with DATA_DIR={d}")
    try:
        tree = json.loads(subprocess.run(["lsblk", "-J", "-b", "-o", "NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT"], capture_output=True, text=True, timeout=10).stdout)
    except Exception:
        return

    def walk(devs):
        for b in devs:
            kids = b.get("children") or []
            used = b.get("fstype") or b.get("mountpoint") or kids
            if b.get("type") == "disk" and not used and int(b.get("size") or 0) > 50 * GB:
                print(f"/dev/{b['name']}: {int(b['size']) / GB:.0f} GB disk with no filesystem; it needs formatting and mounting (README)")
            walk(kids)
    walk(tree.get("blockdevices", []))


def model_gb(model, hf_home, python):
    size = None
    code = ("import sys,huggingface_hub as h;i=h.HfApi().model_info(sys.argv[1],files_metadata=True);"
            "print(sum(s.size or 0 for s in i.siblings))")
    try:
        r = subprocess.run([python, "-c", code, model], capture_output=True, text=True, timeout=60)
        size = int(r.stdout.strip()) if r.returncode == 0 else None
    except Exception:
        pass
    size = size if size else 31 * GB
    have = 0
    blobs = os.path.join(hf_home, "hub", "models--" + model.replace("/", "--"), "blobs")
    if os.path.isdir(blobs):
        for f in os.listdir(blobs):
            try:
                have += os.path.getsize(os.path.join(blobs, f))
            except OSError:
                pass
    print(f"{size / GB:.1f} {max(0, size - have) / GB:.1f}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pick"
    if cmd == "pick":
        print(pick())
    elif cmd == "free":
        print(f"{free_gb(sys.argv[2]):.0f}")
    elif cmd == "bigger":
        bigger(sys.argv[2])
    elif cmd == "model-gb":
        model_gb(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else sys.executable)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
