#!/bin/sh

# path to openSMILE 2.3 SMILExtract
openSMILE=/home/taufiq/Downloads/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract

# ComParE 2016 configuration file - included in the openSMILE 2.3 package
configFile=/home/taufiq/Downloads/opensmile-2.3.0/config/ComParE_2016.conf

labels_file=../lab/ComParE2018_Heartbeat.tsv

feat_dir=../arff
mkdir -p $feat_dir

feat_name=ComParE2018_Heartbeat.ComParE
audio_dir=../wav

train_lld_arff=$feat_dir/$feat_name.train.lld.arff
devel_lld_arff=$feat_dir/$feat_name.devel.lld.arff
test_lld_arff=$feat_dir/$feat_name.test.lld.arff


# Extract LLDs for train & devel
rm -f $train_lld_arff
rm -f $devel_lld_arff

while read line; do
    wavefile=$(echo $line | sed -e 's/\s.*$//')
    if ! echo "$wavefile" | grep -q ".wav"; then
        continue
    fi
    
    # Determine label (if no test set)
    label=$(echo $line | sed -n '${s/.* //; p}')
    if echo "$wavefile" | grep -q "train"; then
        lld_arff=$train_lld_arff
    elif echo "$wavefile" | grep -q "devel"; then
        lld_arff=$devel_lld_arff
    else
        echo "Unknown partition!"
    fi
    
    "$openSMILE" -C "$configFile" -appendcsvlld 1 -instname "$wavefile" -I "$audio_dir/$wavefile" -lldarffoutput "$lld_arff" -arfftargetsfile arff_targets.conf.inc -class "$label"
done < $labels_file


# Extract LLDs for test
#rm -f $test_lld_arff

#label=?
#lld_arff=$test_lld_arff

#for wavefile in $audio_dir/test*.wav; do
#    instname=${wavefile##*/}
#    "$openSMILE" -C "$configFile" -appendcsvlld 1 -instname "$instname" -I "$audio_dir/$wavefile" -#lldarffoutput "$lld_arff" -arfftargetsfile arff_targets.conf.inc -class "$label"
#done
