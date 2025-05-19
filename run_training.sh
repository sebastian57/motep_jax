#!/bin/bash

set -e 
PYTHON_EXE="python3" 
SCRIPT_PATH="./motep_jax_train.py" 

TRAINING_CFG="training.cfg" 
LEVEL=2              
STEPS1=2000                
STEPS2=1000                
STEPS3=100                 
SPECIES=""
PKL_FILE="jax_images_data"  
NAME="timing_results_level_${LEVEL}" 
TRAIN1="true"
TRAIN2="false"
TRAIN3="false"

CMD="$PYTHON_EXE $SCRIPT_PATH"

CMD+=" --training_cfg \"$TRAINING_CFG\""
CMD+=" --level $LEVEL"
CMD+=" --steps1 $STEPS1"
CMD+=" --steps2 $STEPS2" 
CMD+=" --steps3 $STEPS3" 
if [[ -n "$SPECIES" ]]; then
    CMD+=" --species \"$SPECIES\""
fi
CMD+=" --pkl_file \"$PKL_FILE\""
CMD+=" --name \"$NAME\""
CMD+=" --train1 $TRAIN1"
CMD+=" --train2 $TRAIN2"
CMD+=" --train3 $TRAIN3"

echo "  Python Script: $SCRIPT_PATH"
echo "  Config File:   $TRAINING_CFG"
echo "  Level:         $LEVEL"
echo "  Steps1:        $STEPS1"
echo "  Steps2:        $STEPS2"
echo "  Steps3:        $STEPS3"
echo "  Species:       ${SPECIES:-<Default/None>}" 
echo "  PKL File:      $PKL_FILE"
echo "  Output Name:   $NAME (will produce training_results/${NAME}_coeffs.txt)"
echo "-----------------------------------------------------"
echo "Executing command:"
echo "$CMD" 
echo "-----------------------------------------------------"

eval $CMD 

echo "-----------------------------------------------------"
echo "Script finished."