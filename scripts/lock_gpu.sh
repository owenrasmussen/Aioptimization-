#!/usr/bin/env bash
# Lock GPU clocks and power limit so runs are comparable. Needs root/admin.
# Usage:
#   sudo scripts/lock_gpu.sh lock [core_mhz] [power_w]   (defaults: 1500 MHz, 300 W)
#   sudo scripts/lock_gpu.sh unlock
#   scripts/lock_gpu.sh status
# Pick a core clock your card can hold under sustained load without hitting the
# power or thermal limit; check with `status` during a long llama-bench run.
set -euo pipefail

GPU="${GPU_INDEX:-0}"
CMD="${1:-status}"

case "$CMD" in
  lock)
    CLK="${2:-1500}"
    PWR="${3:-300}"
    nvidia-smi -i "$GPU" -pm 1 || true   # persistence mode (Linux only)
    nvidia-smi -i "$GPU" -pl "$PWR"
    nvidia-smi -i "$GPU" -lgc "$CLK,$CLK"
    echo "Locked GPU $GPU: core ${CLK} MHz, power ${PWR} W"
    ;;
  unlock)
    nvidia-smi -i "$GPU" -rgc
    nvidia-smi -i "$GPU" -pl "$(nvidia-smi -i "$GPU" --query-gpu=power.default_limit --format=csv,noheader,nounits)"
    echo "Reset clocks and power limit on GPU $GPU"
    ;;
  status)
    nvidia-smi -i "$GPU" --query-gpu=name,driver_version,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used \
      --format=csv
    ;;
  *)
    echo "unknown command: $CMD" >&2; exit 1 ;;
esac
