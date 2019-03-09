#!/bin/sh

model_dir=../../models/
result_csv=../../results_2class.csv

run_name="None"
epoch="None"
val_acc="None"

# Function to retrieve last run and epoch number and validation value

retrieve_best () {
    local best_epoch
    while IFS=, read -r _ _ _ val _ _ _ _ epoch _ filename _
        do
            run_name=$filename
            best_epoch=$epoch
            val_acc=$val
        done < "$1"
    val_acc=$(printf "%.4f" $(bc -l <<< "scale=5; ($val_acc)/100"))
    epoch=$(printf "%04d" $(( best_epoch+1 )))

    if [ ! -f "$model_dir/$run_name/weights.$epoch-$val_acc.hdf5" ]; then
        echo "Warning: Weights file does not exist!"
    fi
}

for type in 'zero' 1 2 3 4 'gamma'
do
    for fold in "fold0" #"fold1" "fold2" "fold3"
    do

#type=3
#fold="fold1"
        batch_size=64
        # Training with smaller batch_size
        if [ ! $type == "zero" ]; then

            echo "FOLD: $fold STAGE: 1 BS: $batch_size"
            python heartnet_v2.py "$fold"_noFIR --batch_size "$batch_size" --epochs\
             300 --type $type --comment "FIR $type DoubleBalanced Train-a"

        fi

        # Re-training with larger batch_size
#        batch_size=1024
        echo "FOLD: $fold STAGE: 2 BS: $batch_size"
        retrieve_best $result_csv
        python heartnet_v2.py "$fold"_noFIR --batch_size "$batch_size" --epochs\
         450 --loadmodel "$model_dir/$run_name/weights.$epoch-$val_acc.hdf5"\
         --type $type --comment "FIR $type stage 2 " --lr 0.000009843784
    done
done
