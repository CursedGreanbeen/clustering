#!/usr/bin/env bash
# Full clustering + evaluation chain, sequential over one or more complexes.
#
# Usage (from anywhere):
#   ./run_clustering.sh [-w N] [pdb_id ...]     # no pdb args -> all complexes in 500-seeds
#   ./run_clustering.sh -w 8                    # all complexes, 8 parser workers
#   ./run_clustering.sh 8jow 8ker               # just these two
#
# Per complex: 1 parser(multi) 2 matrix 3 cluster 4 dockq_conf 5 parser(single) 6 knn
#               7 collect 8 analysis 9 plots_upd 10 plots 11 build_report.
# Then, once for the whole set: build_index (overview page; runs even if some complexes failed).
# A failed step aborts that complex (later steps depend on it) but not the whole run.
# Outputs (npz/npy/tsv/png) go to /mnt/f0771a75-.../clustering; HTML reports to
# /mnt/1857e392-.../results/reports/html; logs to /tmp/clustering_logs.
set -u

PY=/opt/conda/bin/python3.10
ROOT500=/mnt/f0771a75-815c-4553-ba74-9aa70f57f995/500-seeds
ROOTSINGLE=/mnt/f0771a75-815c-4553-ba74-9aa70f57f995/single-seed
TSV=/mnt/1857e392-c689-44a8-9c3e-e7f9beab82f6/results/reports/anarci_residues.tsv
REPORTS=/mnt/1857e392-c689-44a8-9c3e-e7f9beab82f6/results/reports
REF=$REPORTS/anarci_residues_ref.tsv
DOCKQ_MULTI=${DOCKQ_MULTI:-$REPORTS/dockq_500seeds_75comp_all5_interfaces_20261019.tsv}
DOCKQ_SINGLE=${DOCKQ_SINGLE:-$REPORTS/dockq_singleseed_75comp_all5_interfaces_20261019.tsv}
export DOCKQ_500="$DOCKQ_MULTI" DOCKQ_SINGLE   # keep python steps in sync (dockq_conf.py)
LOG_DIR=/tmp/clustering_logs
WORKERS=1
HTML_DIR=/mnt/1857e392-c689-44a8-9c3e-e7f9beab82f6/results/reports/html
STEP_TIMEOUT=2h      # per step; a hang becomes a normal FAIL (exit 124)
FORCE=0              # -f: rebuild complexes that already have a report

while [[ "${1:-}" == -* ]]; do
  case "$1" in
    -w) [[ -n "${2:-}" ]] || { echo "-w needs a number" >&2; exit 2; }; WORKERS="$2"; shift 2 ;;
    -t) [[ -n "${2:-}" ]] || { echo "-t needs a value, e.g. 2h" >&2; exit 2; }; STEP_TIMEOUT="$2"; shift 2 ;;
    -f) FORCE=1; shift ;;
    -h) grep '^#' "$0" | grep -v '^#!/'; exit 0 ;;
    *)  echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ $# -gt 0 ]]; then PDBS=("$@"); else mapfile -t PDBS < <(cd "$ROOT500" && for d in */; do compgen -G "${d}seed-*_sample-*" >/dev/null && echo "${d%/}"; done); fi
mkdir -p "$LOG_DIR"
cd "$(dirname "$0")"   # imports in the python files assume this cwd

declare -A FAILED SKIPPED WARN
step() {  # step <label> <cmd...> — echo, run (output to terminal + log), check exit code
  local label="$1" rc; shift
  echo "== $label =="
  set -o pipefail
  timeout -k 30 "$STEP_TIMEOUT" "$@" 2>&1 | tee -a "$LOG"; rc=$?
  set +o pipefail
  if [[ $rc -eq 0 ]]; then echo "   OK: $label"; return 0; fi
  echo "   FAIL: $label (exit $rc, log: $LOG)"; return $rc
}
soft() {  # like step, but a failure only warns and the chain continues
  if ! step "$@"; then WARN[$pdb]+="${1%% *} "; fi   # records e.g. "4/11"
  return 0
}
precheck() {  # precheck <pdb>: prints every problem found, returns 1 if any
  local p="$1" bad=0 f
  compgen -G "$ROOT500/$p/seed-*_sample-*" >/dev/null    || { echo "   no models in $ROOT500/$p"; bad=1; }
  compgen -G "$ROOTSINGLE/$p/seed-*_sample-*" >/dev/null || { echo "   no models in $ROOTSINGLE/$p"; bad=1; }
  for f in "$TSV" "$REF" "$DOCKQ_MULTI" "$DOCKQ_SINGLE"; do
    [[ -r "$f" ]] || { echo "   cannot read $f"; bad=1; continue; }
    grep -qw -m1 -- "$p" "$f" || { echo "   $p not found in $f"; bad=1; }
  done
  return $bad
}

for pdb in "${PDBS[@]}"; do
  if [[ $FORCE -eq 0 && -f "$HTML_DIR/$pdb.html" ]]; then
    echo "########## $pdb SKIPPED (report exists, use -f to rebuild) ##########"
    SKIPPED[$pdb]=1
    continue
  fi
  rm -f "$HTML_DIR/$pdb.html"    # a failed rerun must not leave an old report that looks finished
  LOG="$LOG_DIR/${pdb}.log"
  : > "$LOG"
  echo "########## $pdb ##########"
  echo "== 0/11 precheck $pdb =="
  out=$(precheck "$pdb"); rc=$?
  [[ -z "$out" ]] || echo "$out" | tee -a "$LOG"
  if [[ $rc -ne 0 ]]; then
    echo "   FAIL: precheck (see above)"
    FAILED[$pdb]=1
    echo "########## $pdb FAILED ##########"
    continue
  fi
  ok=1
  step "1/11 parser multi $pdb"  "$PY" -c "from parser import parse_complex; parse_complex('$ROOT500', '$pdb', '$TSV', workers=$WORKERS)" && \
  step "2/11 matrix $pdb"        "$PY" -c "from matrix import matrix_A; matrix_A('$pdb', '$TSV')" && \
  step "3/11 cluster $pdb"       "$PY" -c "from cluster import cluster_complex; cluster_complex('$pdb')" && \
  soft "4/11 dockq_conf $pdb (non-blocking)" "$PY" -c "from dockq_conf import evaluate; evaluate('$pdb')" && \
  step "5/11 parser single $pdb" "$PY" -c "from parser import parse_complex; parse_complex('$ROOTSINGLE', '$pdb', '$TSV', run='single')" && \
  soft "6/11 knn $pdb (non-blocking)" "$PY" -c "from knn_assign import assign_single; assign_single('$pdb')" && \
  step "7/11 collect $pdb"       "$PY" collect.py "$pdb" && \
  step "8/11 analysis $pdb"      "$PY" analysis.py "$pdb" && \
  step "9/11 plots_upd $pdb"     "$PY" plots_upd.py "$pdb" && \
  soft "10/11 plots preview $pdb (non-blocking)" "$PY" plots.py "$pdb" && \
  step "11/11 report $pdb"       "$PY" build_report.py "$pdb" || ok=0
  [[ $ok -eq 1 ]] || FAILED[$pdb]=1
  echo "########## $pdb $([[ $ok -eq 1 ]] && echo OK || echo FAILED) ##########"
done

LOG="$LOG_DIR/index.log"; : > "$LOG"   # index covers every stats_*.json on disk, incl. earlier runs
step "index build_index" "$PY" build_index.py || INDEX_OK=0

echo "===== summary ====="
fail=0
for pdb in "${PDBS[@]}"; do
  if   [[ -n "${FAILED[$pdb]:-}"  ]]; then echo "  FAIL  $pdb  (log: $LOG_DIR/${pdb}.log)"; fail=1
  elif [[ -n "${SKIPPED[$pdb]:-}" ]]; then echo "  SKIP  $pdb  (report already exists)"
  elif [[ -n "${WARN[$pdb]:-}"    ]]; then echo "  WARN  $pdb  (non-blocking steps failed: ${WARN[$pdb]}; log: $LOG_DIR/${pdb}.log)"
  else echo "  OK    $pdb"; fi
done
[[ "${INDEX_OK:-1}" -eq 1 ]] || { echo "  FAIL  index  (log: $LOG)"; fail=1; }
exit $fail
