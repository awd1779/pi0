#!/bin/bash
# Debug script for eggplant task
cd /home/ubuntu/open-pi-zero
./scripts/clutter_eval/run_paired_eval.sh --task widowx_put_eggplant_in_basket -e ${1:-1} -r ${2:-1}
