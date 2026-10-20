#!/bin/zsh
# Wait for XOR WHITE sweep-1 and sweep-5, then dump their logs.
#
# Verifies whether the run mounted the post-fix core by grepping the log for the
# tuning lines added in the eval_common rewrite
# ("tune PatternLocal k_ratio=", "tune analytical-IG  k=").
set -u
cd /Users/arif/Research/patternlocal-reproduce
export KAGGLE_API_TOKEN="$(grep KAGGLE_API_TOKEN .env | cut -d= -f2)"
KAG=./.venv/bin/kaggle
OUT=/tmp/xw_watch
mkdir -p "$OUT"

state() {
  $KAG kernels status "mdariffaysalnayem/patternlocal-repro-$1" 2>/dev/null \
    | grep -oE 'KernelWorkerStatus\.[A-Z]+' | tail -1
}

for sw in xor-white-sweep-1 xor-white-sweep-5 xor-corr-sweep-1 xor-corr-sweep-5; do
  while true; do
    st=$(state "$sw")
    case "$st" in
      KernelWorkerStatus.COMPLETE|KernelWorkerStatus.ERROR) break;;
    esac
    sleep 120
  done
  echo "[$(date +%H:%M:%S)] $sw -> $st"
  d="$OUT/$sw"; mkdir -p "$d"
  $KAG kernels output "mdariffaysalnayem/patternlocal-repro-$sw" -p "$d" \
    --file-pattern '.*\.(log|json)$' 2>&1 | tail -2
done
echo "[$(date +%H:%M:%S)] WATCH DONE"
