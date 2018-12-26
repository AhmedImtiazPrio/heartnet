#!/bin/sh

jvm_mem=15000m

### File Paths ###

model_dir=./models/
mkdir -p $model_dir

featfile=../mfcc.csv
labelsfile=../labels.csv

### BOAW Parameters ###

cdbk_size="4096"
#test -z "$cdbk_size" && cdbk_range="64 128" #128 256 512 1024 2048 4096 8192 16384"

multiassign=5

#for cdbk_size in $cdbk_range; do
    outputfile=./feat/boawFeat.$cdbk_size.arff
    rm -f "$outputfile"
    java -Xmx$jvm_mem -jar -XX:-UseGCOverheadLimit openXBOW.jar -i "$featfile" -l "$labelsfile" -o "$outputfile" -standardizeInput -norm 1 -log -size "$cdbk_size" -a "$multiassign" -B "$model_dir/$cdbk_size.codebook"
#done
