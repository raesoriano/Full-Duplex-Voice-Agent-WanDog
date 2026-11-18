#!/usr/bin/env bash
# Run the full-duplex benchmark against WanDog and keep the results.
#
#   bench/run_bench.sh smoke     # dev dataset, 1 round   (~7 min)
#   bench/run_bench.sh full      # 280-clip dataset, 1 round
#   bench/run_bench.sh report    # print the last full run
#
# Everything is written under fd-benchmark/runs/<student>/<timestamp>/ and is
# git-ignored (the submodule's runs/ is). bench/results/ keeps the copies that
# are committed.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="$REPO/fd-benchmark"
PY="$REPO/.venv/bin/python"
MODE="${1:-smoke}"

# The adapter is outside fd-benchmark on purpose: the submodule is pinned
# upstream and its own assistant_entry.py is git-ignored.
export FDBENCH_ENTRY="$REPO/bench/assistant_entry.py"

# fdbench starts the assistant itself and talks to it over this port.
ASSISTANT_CMD="env FDBENCH_ENTRY=$FDBENCH_ENTRY $PY $BENCH/launch_assistant.py --url ws://127.0.0.1:8791 --once"

cd "$BENCH"

case "$MODE" in
  smoke)
    exec "$PY" -m fdbench run \
      --datasets datasets-dev --student wandog-dev \
      --rounds 1 --phrases 1 --runs 1 \
      --connect direct --where local --project "$REPO" \
      --assistant-cmd "$ASSISTANT_CMD" \
      --no-asr --recordings d --yes
    ;;
  full)
    # --no-asr: skip faster-whisper scoring of the assistant's audio (the
    # judge is not configured either). Timing, turn-taking and the stage
    # latencies are still measured, and those are the numbers the deck needs.
    exec "$PY" -m fdbench run \
      --datasets datasets --student wandog \
      --rounds 1 --phrases 10 --runs 3 \
      --connect direct --where local --project "$REPO" \
      --assistant-cmd "$ASSISTANT_CMD" \
      --no-asr --recordings d --yes
    ;;
  report)
    last="$(ls -1dt runs/wandog/*/ 2>/dev/null | head -1)"
    [ -n "$last" ] || { echo "no run found under $BENCH/runs/wandog/" >&2; exit 1; }
    exec "$PY" -m fdbench report "$last"
    ;;
  *)
    echo "usage: $0 [smoke|full|report]" >&2
    exit 2
    ;;
esac
