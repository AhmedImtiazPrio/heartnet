#!/bin/sh

jvm_mem=10000m

### File Paths ###

model_dir=./models/
mkdir -p $model_dir

featfile=../mfcc.csv
labelsfile=../labels.csv
outputfile=../boawFeat.arff

### BOAW Parameters ###

cdbk_size="64,128"
multiassign=5

rm -f boawFeat.arff

java -jar -XX:-UseGCOverheadLimit openXBOW.jar -i "$featfile" -l "$labelsfile" -o "$outputfile" -standardizeInput -norm 1 -log -size "$cdbk_size" -a "$multiassign" -B "$model_dir/codebook"

# -Xmx$jvm_mem
