#!/bin/bash
# Debug script for cube task
cd /home/ubuntu/open-pi-zero
./scripts/clutter_eval/run_paired_eval.sh --task widowx_stack_cube -e ${1:-1} -r ${2:-1}
