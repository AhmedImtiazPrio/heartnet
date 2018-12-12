#!/bin/sh

jvm_mem=15000m

### File Paths ###

model_dir=./models/
mkdir -p $model_dir

fold_name="fold1"
mfcc_path="/media/taufiq/Data1/heart_sound/feature/mfcc/"

trainfile=$mfcc_path/$fold_name.train.mfcc.csv
trainlabels=$mfcc_path/$fold_name.train.labels.csv
devfile=$mfcc_path/$fold_name.dev.mfcc.csv
devlabels=$mfcc_path/$fold_name.dev.labels.csv

### BOAW Parameters ###

cdbk_size="4096"
#test -z "$cdbk_size" && cdbk_range="64 128" #128 256 512 1024 2048 4096 8192 16384"

multiassign=5

## Generating codebook and features for trainset

    outputfile=./feat/$fold_name.train.$cdbk_size.arff
    rm -f "$outputfile"
    java -Xmx$jvm_mem -jar -XX:-UseGCOverheadLimit openXBOW.jar -i "$trainfile" -l "$trainlabels" -o "$outputfile" -standardizeInput -norm 1 -log -size "$cdbk_size" -a "$multiassign" -B "$model_dir/$fold_name.$cdbk_size.codebook"


## Generating features for devset using previous codebook

    outputfile=./feat/$fold_name.dev.$cdbk_size.arff
    rm -f "$outputfile"
    java -Xmx$jvm_mem -jar -XX:-UseGCOverheadLimit openXBOW.jar -i "$devfile" -l "$devlabels" -o "$outputfile" -standardizeInput -norm 1 -log -a "$multiassign" -b "$model_dir/$fold_name.$cdbk_size.codebook"

