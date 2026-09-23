#!/usr/bin/env bash
# Waits for every currently-running (or later-started) build_a6_classification_cache.py
# process to exit, then launches scripts/build_a7_cache.py for all three views.
#
# Written because A6's classification cache build was running as *two*
# separate, overlapping `--workers 25` invocations (50 worker processes on
# this 48-core machine) when this was set up - rather than track specific
# PIDs (fragile if that ever changes), this just polls for "is anything
# matching build_a6_classification_cache.py still running at all", which is
# correct regardless of how many overlapping invocations exist.
#
# Meant to be launched fully detached (setsid + nohup + disown) so it
# survives the launching terminal/session closing - see the one-liner in
# this script's own header comment below for how it was started.
#
#   setsid nohup bash scripts/wait_a6_then_build_a7.sh \
#       > runs/wait_a6_then_build_a7.log 2>&1 < /dev/null &
#   disown
#
# Progress/status: tail -f runs/wait_a6_then_build_a7.log
# A7's own build output goes to the same log, appended after the wait ends.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."  # repo root, regardless of caller's cwd

PYTHON="/home/oriol@newcefe.newage.fr/.conda/envs/torch/bin/python"
A7_WORKERS=25
POLL_SECONDS=120

log() {
    echo "[$(date -Is)] $*"
}

log "wait_a6_then_build_a7: watching for build_a6_classification_cache.py to finish"
log "PID of this watcher: $$"

while pgrep -f "build_a6_classification_cache\.py" > /dev/null; do
    n=$(pgrep -f "build_a6_classification_cache\.py" | wc -l)
    log "still running ($n matching processes) - checking again in ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
done

log "no build_a6_classification_cache.py process found - A6 classification build is done"
log "starting: $PYTHON scripts/build_a7_cache.py --workers $A7_WORKERS"

"$PYTHON" scripts/build_a7_cache.py --workers "$A7_WORKERS"
status=$?

log "build_a7_cache.py exited with status $status"
