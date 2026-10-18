#!/usr/bin/env bash
# Everything a bare Ubuntu 24.04 GPU VM needs before `deploy/qwen.sh start`, in one command that is
# safe to run again (each step skips what is already there). Run sheet: deploy/README.md.
#
# On a fresh VM, before there is a checkout:
#   curl -fsSLo /tmp/fresh.sh https://raw.githubusercontent.com/OliverVillson/llmap/claude/centipede-demo/mugge/deploy/fresh.sh
#   bash /tmp/fresh.sh
# From a checkout:  deploy/qwen.sh fresh
#
#  1. apt: python3-dev (Triton's C helpers), build-essential (gcc, and g++ for nvcc), ninja-build
#     (FlashInfer's kernel builds), git, curl, unzip (the bun installer), python-is-python3 and
#     python3-pytest (the agents' checks)
#  2. the data disk: mounts the disk labelled "data" at /data (and adds it to /etc/fstab, nofail).
#     With no such disk it lists the empty ones and stops; it formats one only when told which:
#     bash fresh.sh --format /dev/vdc   (erases that disk; refused if it has a filesystem or partitions)
#  3. the checkout at /data/llmap (cloned, or brought to origin/$BRANCH), with ~/llmap pointing at it
#  4. uv and bun, linked into /usr/local/bin so they are on PATH in every shell, ssh commands too
#  5. deploy/qwen.sh setup (the venv on the data disk, vLLM, nvcc matched to the CUDA runtime),
#     download (reuses weights already on the disk), bun install, then deploy/qwen.sh preflight
#
# Settings: BRANCH (claude/centipede-demo), REPO_URL, DATA_MNT (/data), BUN_VERSION, UV_VERSION, SKIP_DOWNLOAD=1,
# NO_GPU=1 (skip the driver check and the GPU part of preflight: for testing in a container).
set -euo pipefail

BRANCH="${BRANCH:-claude/centipede-demo}"
REPO_URL="${REPO_URL:-https://github.com/OliverVillson/llmap.git}"
DATA_MNT="${DATA_MNT:-/data}"
BUN_VERSION="${BUN_VERSION:-1.4.2}"
UV_VERSION="${UV_VERSION:-0.11.32}"
FORMAT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --format) FORMAT="${2:-}"; shift 2 || { echo "fresh.sh: --format needs a device, e.g. --format /dev/vdc" >&2; exit 1; } ;;
    *) echo "fresh.sh: unknown argument $1 (only --format /dev/XXX)" >&2; exit 1 ;;
  esac
done

step() { printf '\n== %s\n' "$*"; }
die() { printf '\nfresh.sh: %s\n' "$*" >&2; exit 1; }
SUDO=""; [ "$(id -u)" = 0 ] || SUDO=sudo
ME="$(id -un)"; MYGROUP="$(id -gn)"
export PATH="$HOME/.local/bin:$HOME/.bun/bin:$PATH"

[ -r /etc/os-release ] && . /etc/os-release
[ "${ID:-}" = ubuntu ] || echo "warning: written for Ubuntu 24.04; this is ${PRETTY_NAME:-unknown}" >&2
if [ -n "$SUDO" ] && ! sudo -n true 2>/dev/null; then echo "sudo will ask for your password once"; sudo -v || die "this needs sudo"; fi

# ---------------------------------------------------------------------------------------------
step "1/5 system packages"
# python-is-python3 and python3-pytest: the agents' checks run on this machine, and a plan that says
# `python` or `pytest` would otherwise fail every ticket with "command not found"
PKGS=(python3 python3-dev python-is-python3 python3-pytest build-essential ninja-build git curl ca-certificates unzip)
missing=()
for p in "${PKGS[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p"); done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "installing ${missing[*]}"
  # a fresh VM runs unattended-upgrades in its first minutes and holds apt's locks; apt-get stops at
  # once on a held lock, so wait for it (install has a lock timeout of its own, update does not)
  for i in $(seq 1 90); do
    out="$($SUDO apt-get update -qq 2>&1)" && break
    case "$out" in *"Could not get lock"*|*"Unable to lock"*) ;; *) printf '%s\n' "$out" >&2; die "apt-get update failed: sudo apt-get update" ;; esac
    [ "$i" != 1 ] || echo "another apt is running (unattended-upgrades on a new VM); waiting for it, up to 15 minutes"
    [ "$i" != 90 ] || die "apt is still locked after 15 minutes: ps aux | grep -E 'apt|dpkg|unattended'"
    sleep 10
  done
  $SUDO env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a NEEDRESTART_SUSPEND=1 \
    apt-get install -y -qq -o DPkg::Lock::Timeout=900 "${missing[@]}" >/dev/null </dev/null ||
    die "apt-get install failed: sudo apt-get install -y ${missing[*]}"
fi
# the headers of the python3 that the venv is made from (python3-dev pulls the matching python3.X-dev)
py_h="$(python3 -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["include"], "Python.h"))')"
[ -f "$py_h" ] || die "no $py_h: sudo apt-get install -y python3-dev"
for t in gcc g++ ninja git curl unzip; do command -v "$t" >/dev/null || die "$t is missing after apt: sudo apt-get install -y ${PKGS[*]}"; done
echo "ok: $(python3 --version), $(gcc --version | head -1), ninja $(ninja --version), $(git --version)"

if [ -z "${NO_GPU:-}" ]; then
  if ! command -v nvidia-smi >/dev/null; then
    die "no NVIDIA driver (nvidia-smi). Create the VM from evroc's GPU image, or install one: sudo ubuntu-drivers install --gpgpu && sudo reboot"
  elif ! smi="$(nvidia-smi -L 2>&1)"; then
    # unattended-upgrades on a new VM can update the driver's files under the loaded kernel module
    case "$smi" in *mismatch*) die "the NVIDIA driver was updated under the running kernel module (nvidia-smi: $smi). sudo reboot, then run this again" ;; esac
    die "nvidia-smi fails: $smi. Try: sudo reboot; else sudo ubuntu-drivers install --gpgpu && sudo reboot"
  fi
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# ---------------------------------------------------------------------------------------------
step "2/5 data disk at $DATA_MNT"
GB=1000000000
mounted_at() { findmnt -n -o TARGET --source "$1" 2>/dev/null | head -1 || true; }
# A disk is "empty" only when every probe positively says so; a probe that errors counts as "not
# empty". Whole disk, no partitions, not mounted, not the disk under / or $HOME, no filesystem or
# partition-table signature (wipefs and blkid both ran and found none), first MiB all zeros.
empty_disk() {
  local d="$1" root
  case "$(lsblk -dn -o TYPE "$d" 2>/dev/null)" in disk|loop) ;; *) return 1 ;; esac
  [ -z "$(lsblk -n -o NAME "$d" 2>/dev/null | tail -n +2)" ] || return 1
  [ -z "$(mounted_at "$d")" ] && ! grep -q "^$d " /proc/mounts || return 1
  for root in / "$HOME"; do
    root="$(findmnt -n -o SOURCE --target "$root" 2>/dev/null || true)"
    [ "$root" != "$d" ] && [ "/dev/$(lsblk -n -o PKNAME "$root" 2>/dev/null | head -1)" != "$d" ] || return 1
  done
  local out rc
  out="$($SUDO wipefs -n "$d" 2>&1)" && [ -z "$out" ] || return 1
  rc=0; $SUDO blkid -p "$d" >/dev/null 2>&1 || rc=$?
  [ "$rc" = 2 ] || return 1
  [ "$($SUDO dd if="$d" bs=1M count=1 status=none 2>/dev/null | tr -d '\0' | wc -c)" = 0 ] &&
    [ "$($SUDO dd if="$d" bs=1M count=1 status=none 2>/dev/null | wc -c)" = 1048576 ]
}
dev="$(blkid -L data 2>/dev/null || $SUDO blkid -L data 2>/dev/null || true)"
if [ -z "$dev" ] && [ -n "$FORMAT" ]; then
  [ -b "$FORMAT" ] || die "--format $FORMAT: no such block device (lsblk lists them)"
  empty_disk "$FORMAT" || die "--format $FORMAT: refusing to erase it: it is not a whole, unmounted, blank disk (partitions, a filesystem, data in its first MiB, or it could not be checked). lsblk -f $FORMAT"
  echo "formatting $FORMAT ($(( $(lsblk -dbn -o SIZE "$FORMAT") / GB )) GB) as ext4, label data"
  $SUDO mkfs.ext4 -q -L data "$FORMAT" || die "mkfs.ext4 failed on $FORMAT"
  $SUDO udevadm settle 2>/dev/null || true
  dev="$(blkid -L data 2>/dev/null || $SUDO blkid -L data 2>/dev/null || echo "$FORMAT")"
fi
if [ -n "$dev" ]; then
  at="$(mounted_at "$dev")"
  if [ -z "$at" ]; then
    $SUDO mkdir -p "$DATA_MNT"
    $SUDO mount "$dev" "$DATA_MNT" || die "could not mount $dev at $DATA_MNT: sudo mount $dev $DATA_MNT (then dmesg | tail)"
    at="$DATA_MNT"
  fi
  [ "$at" = "$DATA_MNT" ] || { echo "the data disk $dev is already mounted at $at; using it"; DATA_MNT="$at"; }
  fstype="$(blkid -o value -s TYPE "$dev" 2>/dev/null || $SUDO blkid -o value -s TYPE "$dev")"
  # back after a reboot; nofail so a missing disk never stops the VM from booting
  # (a line of its own for that mount point, e.g. /dev/vdc /data made by hand, counts: two would clash)
  if ! awk -v m="$DATA_MNT" '$1 !~ /^#/ && ($1 == "LABEL=data" || $2 == m) {f = 1} END {exit !f}' /etc/fstab; then
    echo "LABEL=data $DATA_MNT ${fstype:-auto} defaults,nofail 0 2" | $SUDO tee -a /etc/fstab >/dev/null
    echo "added to /etc/fstab: LABEL=data $DATA_MNT"
  fi
elif mountpoint -q "$DATA_MNT"; then
  echo "$DATA_MNT is a mount (not labelled data); using it"
else
  # no data disk: say which disks are empty, and stop rather than fill the small root disk
  empty="$(lsblk -dbn -o NAME,SIZE,TYPE | while read -r n s t; do
    [ "$t" = disk ] && [ "$s" -gt $((50 * GB)) ] && empty_disk "/dev/$n" || continue
    echo "/dev/$n $((s / GB))"; done)"
  root_free=$(( $(df -B1 --output=avail "$HOME" | tail -1) / GB ))
  if [ -n "$empty" ]; then
    echo "no disk labelled data; empty disks (nothing on them):"
    echo "$empty" | while read -r n s; do echo "  $n  $s GB"; done
    first="$(echo "$empty" | head -1 | cut -d' ' -f1)"
    die "to format $first as the data disk (this ERASES it) and carry on, run:  bash ${BASH_SOURCE[0]} --format $first"
  elif [ "$root_free" -ge 60 ]; then
    echo "no data disk, but $HOME has $root_free GB free: using $HOME/qwen-data"
    DATA_MNT=""
  else
    lsblk -o NAME,SIZE,TYPE,FSTYPE,LABEL,MOUNTPOINT >&2
    die "no data disk (labelled data, or an empty one over 50 GB) and only $root_free GB free in $HOME; the weights, venv and caches need about 50 GB. Attach a volume to the VM and run this again"
  fi
fi
if [ -n "$DATA_MNT" ]; then
  DATA_DIR="$DATA_MNT/qwen-data"
  # a dedicated disk: the top folder is ours (a fresh ext4 belongs to root)
  [ -w "$DATA_MNT" ] || $SUDO chown "$ME:$MYGROUP" "$DATA_MNT"
  # a disk kept from an earlier VM can carry another uid; the venv, weights, caches and checkout must be ours
  for d in "$DATA_DIR" "$DATA_MNT/llmap"; do
    if [ -e "$d" ] && [ -n "$(find "$d" -maxdepth 2 ! -user "$ME" -print -quit 2>/dev/null)" ]; then
      echo "taking ownership of $d (made by another user or an earlier VM)"
      $SUDO chown -R "$ME:$MYGROUP" "$d"
    fi
  done
else
  DATA_DIR="$HOME/qwen-data"
fi
mkdir -p "$DATA_DIR" "$HOME/qwen"
touch "$DATA_DIR/.write-test" && rm -f "$DATA_DIR/.write-test" || die "cannot write $DATA_DIR: sudo chown -R $ME:$MYGROUP $DATA_DIR"
echo "$DATA_DIR" > "$HOME/qwen/data.dir"
echo "ok: data folder $DATA_DIR, $(( $(df -B1 --output=avail "$DATA_DIR" | tail -1) / GB )) GB free"

# ---------------------------------------------------------------------------------------------
step "3/5 checkout ($BRANCH)"
if [ -n "$DATA_MNT" ]; then CO="$DATA_MNT/llmap"; else CO="$HOME/llmap"; fi
if [ -L "$HOME/llmap" ] && [ "$(readlink -f "$HOME/llmap")" != "$(readlink -f "$CO")" ]; then rm "$HOME/llmap"; fi
if [ -n "$DATA_MNT" ] && [ -d "$HOME/llmap" ] && [ ! -L "$HOME/llmap" ]; then
  # a checkout on the small root disk: keep it out of the way, the data disk one is used
  old="$HOME/llmap.old-$(date +%Y%m%d-%H%M%S)"; mv "$HOME/llmap" "$old"; echo "moved the checkout on the root disk to $old"
fi
if [ -d "$CO/.git" ]; then
  # from REPO_URL into origin/$BRANCH by name: a checkout cloned single-branch, from another branch,
  # or over ssh (no GitHub key on the VM) would otherwise leave origin/$BRANCH missing or stale
  git -C "$CO" fetch -q "$REPO_URL" "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" ||
    die "git fetch failed in $CO (network?): git -C $CO fetch $REPO_URL $BRANCH"
  if [ -n "$(git -C "$CO" status --porcelain --untracked-files=no)" ]; then
    tag="fresh-$(date +%Y%m%d-%H%M%S)"
    git -C "$CO" stash push -q -m "$tag" && echo "local edits in $CO saved: git -C $CO stash list ($tag)"
  fi
  git -C "$CO" checkout -q -B "$BRANCH" "origin/$BRANCH"
else
  [ ! -e "$CO" ] || die "$CO exists but is not a git checkout; move it away and run this again"
  git clone -q -b "$BRANCH" "$REPO_URL" "$CO" || die "git clone failed: git clone -b $BRANCH $REPO_URL $CO"
fi
[ "$CO" = "$HOME/llmap" ] || ln -sfn "$CO" "$HOME/llmap"
echo "ok: $CO at $(git -C "$CO" log -1 --format='%h %s' | cut -c1-80)$([ "$CO" = "$HOME/llmap" ] || echo ", ~/llmap -> $CO")"
M="$CO/mugge"

# ---------------------------------------------------------------------------------------------
step "4/5 uv and bun"
# pinned release builds straight from GitHub (the same files the official install scripts fetch)
case "$(uname -m)" in x86_64) UV_ARCH=x86_64 BUN_ARCH=x64 ;; aarch64) UV_ARCH=aarch64 BUN_ARCH=aarch64 ;; *) die "unsupported CPU $(uname -m)" ;; esac
if ! [ -x "$HOME/.local/bin/uv" ] || [ "$("$HOME/.local/bin/uv" --version | cut -d' ' -f2)" != "$UV_VERSION" ]; then
  tmp="$(mktemp -d)"; mkdir -p "$HOME/.local/bin"
  curl -fsSL "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$UV_ARCH-unknown-linux-gnu.tar.gz" | tar -xz -C "$tmp" --strip-components 1 &&
    install -m 755 "$tmp/uv" "$tmp/uvx" "$HOME/.local/bin/" ||
    { curl -fsSL https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh >/dev/null; } ||
    die "uv install failed: curl -LsSf https://astral.sh/uv/install.sh | sh"
  rm -rf "$tmp"
fi
if ! [ -x "$HOME/.bun/bin/bun" ] || [ "$("$HOME/.bun/bin/bun" --version)" != "$BUN_VERSION" ]; then
  tmp="$(mktemp -d)"; mkdir -p "$HOME/.bun/bin"
  { curl -fsSLo "$tmp/bun.zip" "https://github.com/oven-sh/bun/releases/download/bun-v$BUN_VERSION/bun-linux-$BUN_ARCH.zip" &&
    unzip -q -o "$tmp/bun.zip" -d "$tmp" && install -m 755 "$tmp/bun-linux-$BUN_ARCH/bun" "$HOME/.bun/bin/bun"; } ||
    { curl -fsSL https://bun.sh/install | bash -s "bun-v$BUN_VERSION" >/dev/null; } ||
    die "bun install failed: curl -fsSL https://bun.sh/install | bash -s bun-v$BUN_VERSION"
  rm -rf "$tmp"
fi
# /usr/local/bin is on PATH for login shells, nohup, and `ssh evroc 'cmd'` alike (~/.bashrc is not read there)
for t in "$HOME/.local/bin/uv" "$HOME/.bun/bin/bun"; do
  [ -x "$t" ] || continue
  [ "$(readlink -f "/usr/local/bin/${t##*/}" 2>/dev/null)" = "$(readlink -f "$t")" ] || $SUDO ln -sfn "$t" "/usr/local/bin/${t##*/}"
done
[ -e /usr/local/bin/bunx ] || [ ! -x "$HOME/.bun/bin/bun" ] || $SUDO ln -sfn "$HOME/.bun/bin/bun" /usr/local/bin/bunx
hash -r
env -i PATH=/usr/local/bin:/usr/bin:/bin bun --version >/dev/null || die "bun is not on the plain PATH: sudo ln -sfn ~/.bun/bin/bun /usr/local/bin/bun"
env -i PATH=/usr/local/bin:/usr/bin:/bin uv --version >/dev/null || die "uv is not on the plain PATH: sudo ln -sfn ~/.local/bin/uv /usr/local/bin/uv"
echo "ok: bun $(bun --version), $(uv --version)"

# ---------------------------------------------------------------------------------------------
step "5/5 vLLM, the weights, the demo"
export DATA_DIR
"$M/deploy/qwen.sh" setup || die "setup stopped (see above). Run the same command again: what is installed is kept"
# 31 GB: an ssh drop or a network error here loses nothing; the same command carries on
[ -n "${SKIP_DOWNLOAD:-}" ] || "$M/deploy/qwen.sh" download ||
  die "the weights download stopped (network, or the Hub). Run the same command again: it resumes where it stopped"
( cd "$M" && bun install --frozen-lockfile >/dev/null 2>&1 || bun install ) || die "bun install failed: cd $M && bun install"
echo
"$M/deploy/qwen.sh" preflight || die "preflight found problems (each FAIL line above has its fix); fix them and run: cd ~/llmap/mugge && deploy/qwen.sh preflight"
cat <<EOF

ready. Next:
  cd ~/llmap/mugge
  CREW_WORKERS=64 deploy/qwen.sh start && deploy/qwen.sh wait && deploy/qwen.sh check
EOF
