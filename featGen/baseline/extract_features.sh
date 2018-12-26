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

train_arff=$feat_dir/$feat_name.train.arff
devel_arff=$feat_dir/$feat_name.devel.arff
test_arff=$feat_dir/$feat_name.test.arff


# Extract ComParE features for train and devel
rm -f $train_arff
rm -f $devel_arff

while read line; do
    wavefile=$(echo $line | sed -e 's/\s.*$//')
    if ! echo "$wavefile" | grep -q ".wav"; then
        continue
    fi
    
    # Determine label (if no test set)
    label=$(echo $line | sed -n '${s/.* //; p}')
    if echo "$wavefile" | grep -q "train"; then
        arff=$train_arff
    elif echo "$wavefile" | grep -q "devel"; then
        arff=$devel_arff
    elif echo "$wavefile" | grep -q "test"; then
        continue
    else
        echo "Unknown partition!"
    fi
    
    "$openSMILE" -C "$configFile" -appendarff 1 -instname "$wavefile" -I "$audio_dir/$wavefile" -output "$arff" -arfftargetsfile arff_targets.conf.inc -class "$label"
done < $labels_file


# Extract ComParE features for test
rm -f $test_arff

label=?
arff=$test_arff

for wavefile in $audio_dir/test*.wav; do
    instname=${wavefile##*/}
    "$openSMILE" -C "$configFile" -appendarff 1 -instname "$instname" -I "$audio_dir/$wavefile" -output "$arff" -arfftargetsfile arff_targets.conf.inc -class "$label"
done
