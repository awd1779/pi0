#!/bin/bash
# Debug script for banana task (novel object - out of distribution)
# Usage: ./debug_banana.sh [episodes] [runs]
cd /home/ubuntu/open-pi-zero
./scripts/clutter_eval/run_paired_eval.sh --task widowx_banana_on_plate -e ${1:-1} -r ${2:-1}
