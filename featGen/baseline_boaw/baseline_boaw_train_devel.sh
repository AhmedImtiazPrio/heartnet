#!/bin/sh

# ComParE 2018, Heartbeat task
# Baseline script: training on training set, results on development set

# set -x

# path to your feature directory (ARFF files)
feat_dir=../arff

# directory where SVM models will be stored
model_dir=./models/train_devel
mkdir -p $model_dir

# directory where evaluation results will be stored
eval_dir=./eval/train_devel
mkdir -p $eval_dir

# feature file basename
feat_name=ComParE2018_Heartbeat.ComParE

# path to Weka's jar file
weka_jar=/tools/weka-3-8-1/weka.jar
test -f $weka_jar || exit -1

# memory to allocate for the JVM
jvm_mem=10000m

# SVM complexity constant
C=$1
test -z "$C" && C=1.0E-3

# openXBOW parameters
openXBOW_arff_labels="0,1,2"
openXBOW_attributes_string="nt1[65]2[65]c"
openXBOW_num_assignments=10
openXBOW_size_codebook="250,250"
lab=$(echo $openXBOW_size_codebook | sed 's/,/+/g')
lab=$(echo "$lab +2" | bc)  # Must be the sum of codebook sizes + 2

#epsilon-intensive loss
L=$2
test -z "$L" && L=0.1

train_lld_arff=$feat_dir/$feat_name.train.lld.arff
train_boaw_arff=$feat_dir/$feat_name.train.BoAW.arff
train_boaw_arff_up=$feat_dir/$feat_name.train.BoAW.upsampled.arff

test_lld_arff=$feat_dir/$feat_name.devel.lld.arff
test_boaw_arff=$feat_dir/$feat_name.devel.BoAW.arff

# Compute BoAW
if [ ! -s "$train_boaw_arff" ]; then
	java -Xmx$jvm_mem -jar openXBOW.jar -writeName -i "$train_lld_arff" -o "$train_boaw_arff" -standardizeInput -log -size "$openXBOW_size_codebook" -B "$model_dir/codebook" -a "$openXBOW_num_assignments" -attributes "$openXBOW_attributes_string" -arffLabels "$openXBOW_arff_labels"
fi
if [ ! -s "$test_boaw_arff" ]; then
	java -Xmx$jvm_mem -jar openXBOW.jar -writeName -i "$test_lld_arff" -o "$test_boaw_arff" -b
"$model_dir/codebook" -attributes "$openXBOW_attributes_string" -arffLabels "$openXBOW_arff_labels"
fi

# Upsampling of train
test -f $train_boaw_arff_up || perl upsample.pl $train_boaw_arff $train_boaw_arff_up "train"


# model file name
svm_model_name=$model_dir/$feat_name.BoAW.train.SMO.C$C.L$L.model

# train SVM using Weka's SMO, using FilteredClassifier wrapper to ignore first attribute (instance name)
if [ ! -s "$svm_model_name" ]; then
      	java -Xmx$jvm_mem -classpath $weka_jar weka.classifiers.meta.FilteredClassifier -v -o -no-cv -c last -t "$train_boaw_arff_up" -d "$svm_model_name" -F "weka.filters.unsupervised.attribute.Remove -R 1" -W weka.classifiers.functions.SMO -- -C $C -L $L -N 1 -M -P 1.0E-12 -V -1 -W 1 -K "weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0" || exit 1
fi

echo "finished train model"

# evaluate SVM and write predictions
pred_file=$eval_dir/$feat_name.BoAW.SMO.C$C.L$L.pred
if [ ! -s "$pred_file" ]; then
	java -Xmx$jvm_mem -classpath $weka_jar weka.classifiers.meta.FilteredClassifier -o -c last -l "$svm_model_name" -T "$test_boaw_arff" -p 0 -distribution > "$pred_file" || exit 1
fi

echo "finished evaluate SVM and write predictions"

# produce ARFF file in submission format
pred_arff=$eval_dir/$feat_name.BoAW.SMO.C$C.L$L.arff
if [ ! -f "$pred_arff" ]; then
	perl format_pred.pl $test_boaw_arff $pred_file $pred_arff $lab || exit 1
fi

echo "Created submission format ARFF: $pred_arff"

ref_arff=$feat_dir/$feat_name.devel.arff  # standard ComParE feature set (no BoAW)
lab_compare=6375
if [ -f "$ref_arff" ]; then
	echo "Found reference ARFF: $ref_arff"
	result_file=$eval_dir/`basename $pred_file .pred`.result
	if [ ! -f $result_file ]; then
		perl score.pl $ref_arff $pred_arff $lab_compare | tee $result_file
	else
		cat $result_file
	fi
fi

