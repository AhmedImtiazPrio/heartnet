#!/bin/sh

### File Paths ###

featfile=../mfcc.csv
labelsfile=../labels.csv
outputfile=../boawFeat.arff

### BOAW Parameters ###

size= 500
multiassign= 5

rm -f boawFeat.arff

java -jar openXBOW.jar -i "$featfile" -l "$labelsfile" -o "$outputfile" -size "$size" -a "$multiassign"
