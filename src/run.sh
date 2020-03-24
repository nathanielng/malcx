#!/bin/bash

N="200"
DATE=`date +%Y%m%d`
ALGO="tpe"
RANDOM_STATE="118956"
OUTPUT_FILE="alcx_${ALGO}_${DATE}.out"
echo "python alcx.py --algorithm ${ALGO} --random_state ${RANDOM_STATE} --evaluations ${N} 2>&1 | tee ${OUTPUT_FILE}"
python alcx.py --algorithm ${ALGO} --random_state ${RANDOM_STATE} --evaluations ${N} 2>&1 | tee ${OUTPUT_FILE}

