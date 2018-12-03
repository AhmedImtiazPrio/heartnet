#!/bin/sh

# path to openSMILE 2.3 SMILExtract
openSMILE=/home/taufiq/Downloads/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract

# ComParE 2016 configuration file - included in the openSMILE 2.3 package
configFile=/home/taufiq/Downloads/opensmile-2.3.0/config/MFCC12_0_D_A.conf

labels_file=../wav.tsv

feat_dir=../mfcc
mkdir -p $feat_dir

feat_name=ComParE2018_Heartbeat.ComParE
audio_dir=../wav

train_mfcc_arff=$feat_dir/$feat_name.train.mfcc
devel_mfcc_arff=$feat_dir/$feat_name.devel.mfcc
test_mfcc_arff=$feat_dir/$feat_name.test.mfcc


# Extract MFCCs for train & devel
rm -f $train_mfcc_arff
rm -f $devel_mfcc_arff

while read line; do
    wavefile=$(echo $line | sed -e 's/\s.*$//')
    if ! echo "$wavefile" | grep -q ".wav"; then
        continue
    fi
    
    # Determine label (if no test set)
    label=$(echo $line | sed -n '${s/.* //; p}')
    if echo "$wavefile" | grep -q "train"; then
        mfcc_arff=$train_mfcc_arff.$wavefile.arff
    elif echo "$wavefile" | grep -q "devel"; then
        mfcc_arff=$devel_mfcc_arff.$wavefile.arff
    else
        echo "Unknown partition!"
    fi
    
    "$openSMILE" -C "$configFile" -I "$audio_dir/$wavefile" -O "$mfcc_arff"
done < $labels_file


# Extract mfccs for test
#rm -f $test_mfcc_arff

#label=?
#mfcc_arff=$test_mfcc_arff

#for wavefile in $audio_dir/test*.wav; do
#    instname=${wavefile##*/}
#    "$openSMILE" -C "$configFile" -appendcsvlld 1 -instname "$instname" -I "$audio_dir/$wavefile" -#lldarffoutput "$mfcc_arff" -arfftargetsfile arff_targets.conf.inc -class "$label"
#done
