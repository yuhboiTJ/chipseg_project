#!/bin/bash
# Launcher for the live microchip area tool. Drop a desktop shortcut that
# calls this script for the double-click experience on the Pi.
#
# One-time setup on the Pi:
#   chmod +x run_chip_tool.sh
#
# Run:
#   ./run_chip_tool.sh
# or pass through any args, e.g.:
#   ./run_chip_tool.sh --device 1 --calibration derived

set -e
cd "$(dirname "$0")"
python3 scripts/live_demo.py "$@"
