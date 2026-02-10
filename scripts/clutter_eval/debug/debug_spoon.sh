#!/bin/bash
# Debug script for spoon task
cd /home/ubuntu/open-pi-zero
./scripts/clutter_eval/run_paired_eval.sh --task widowx_spoon_on_towel -e ${1:-1} -r ${2:-1}
