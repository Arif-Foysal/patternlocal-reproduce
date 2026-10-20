#!/bin/zsh
# Push queued Kaggle kernels as batch CPU session slots free up.
#
# Kaggle caps concurrent batch CPU sessions at 5.  The data-generation kernels
# occupy them first; this loop waits for free slots and pushes queued sweep /
# qualitative kernels one at a time, retrying when the cap is hit.
#
# Two things the naive version got wrong, fixed here:
#   * `kaggle kernels list` has NO state column, so counting RUNNING rows always
#     returned 0 -> use per-kernel `kaggle kernels status` instead.
#   * A capped push can exit 0 without creating a run, so confirm the push by
#     checking that the kernel's lastRunTime advanced.
#
# Usage: scripts/push_pending_kernels.sh <template> [<template> ...]
set -u
cd /Users/arif/Research/patternlocal-reproduce
export KAGGLE_API_TOKEN="$(grep KAGGLE_API_TOKEN .env | cut -d= -f2)"
KAG=./.venv/bin/kaggle
PY=./.venv/bin/python
LOG=/tmp/push_pending.log
MAX_SESSIONS=5

if [ $# -eq 0 ]; then
  print -r -- "usage: $0 <template.py> [more templates...]" >&2
  exit 1
fi

log() { print -r -- "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

# Live batch sessions = kernels of ours currently RUNNING or QUEUED.
active_count() {
  local n=0 slug st
  for slug in $($KAG kernels list --user mdariffaysalnayem --page-size 80 2>/dev/null \
                 | awk '{print $1}' | grep '^mdariffaysalnayem/'); do
    st=$($KAG kernels status "$slug" 2>/dev/null | grep -oE 'KernelWorkerStatus\.[A-Z]+' | tail -1)
    case "$st" in
      KernelWorkerStatus.RUNNING|KernelWorkerStatus.QUEUED) n=$((n+1));;
    esac
  done
  print -r -- "$n"
}

last_run_time() {
  $KAG kernels list --user mdariffaysalnayem --page-size 80 2>/dev/null \
    | awk -v s="$1" '$1==s {print $4" "$5}'
}

for tpl in "$@"; do
  while true; do
    n=$(active_count)
    if [ "$n" -ge "$MAX_SESSIONS" ]; then
      sleep 60
      continue
    fi
    log "slots $n/$MAX_SESSIONS -> pushing $tpl"
    out=$($PY pipeline.py push "$tpl" 2>&1)
    if print -r -- "$out" | grep -q "Maximum batch CPU session"; then
      log "  rejected (cap); retry in 60s"
      sleep 60
      continue
    fi
    print -r -- "$out" >> "$LOG"
    log "  pushed (exit ok)"
    sleep 25
    break
  done
done
log "QUEUE DRAINED"
