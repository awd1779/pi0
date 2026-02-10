#!/bin/bash
# Debug script for carrot task
# Usage: ./debug_carrot.sh [episodes] [runs]
cd /home/ubuntu/open-pi-zero
./scripts/clutter_eval/run_paired_eval.sh --task widowx_carrot_on_plate -e ${1:-1} -r ${2:-1}
