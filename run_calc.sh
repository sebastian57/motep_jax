#!/bin/bash

set -e
PYTHON_EXE="python3"
SCRIPT_PATH="./motep_jax_calc.py"

ATOM_IDS=30

CMD="$PYTHON_EXE $SCRIPT_PATH"
CMD+=" --atom_ids $ATOM_IDS"

echo "Running calc script for $ATOM_IDS configurations:"

eval $CMD 

echo "-----------------------------------------------------"
echo "Timing script finished."

