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
weka_jar=/opt/weka-3-8-2/weka.jar
test -f $weka_jar || exit -1

# memory to allocate for the JVM
jvm_mem=4096m

# SVM complexity constant
C=$1
#test -z "$C" && C_range="1.0E-1 1.0E-2 1.0E-3 1.0E-4 1.0E-5 1.0E-6"
test -z "$C" && C=1.0E-4

#epsilon-intensive loss
L=$2
test -z "$L" && L=0.3

lab=6375

train_arff=$feat_dir/$feat_name.train.arff
train_arff_up=$feat_dir/$feat_name.train.upsampled.arff

test_arff=$feat_dir/$feat_name.devel.arff

# Upsample training set
test -f $train_arff_up || perl upsample.pl $train_arff $train_arff_up "train"

#for C in $C_range; do

	# model file name
	svm_model_name=$model_dir/$feat_name.train.SMO.C$C.L$L.model

	# train SVM using Weka's SMO, using FilteredClassifier wrapper to ignore first attribute (instance name)
	if [ ! -s "$svm_model_name" ]; then
		java -Xmx$jvm_mem -classpath $weka_jar weka.classifiers.meta.FilteredClassifier -v -o -no-cv -c last -t "$train_arff_up" -d "$svm_model_name" -F "weka.filters.unsupervised.attribute.Remove -R 1" -W weka.classifiers.functions.SMO -- -C $C -L $L -N 1 -M -P 1.0E-12 -V -1 -W 1 -K "weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0" || exit 1
	fi

	echo "finished train model"

	# evaluate SVM and write predictions
	pred_file=$eval_dir/$feat_name.SMO.C$C.L$L.pred
	if [ ! -s "$pred_file" ]; then
		java -Xmx$jvm_mem -classpath $weka_jar weka.classifiers.meta.FilteredClassifier -o -c last -l "$svm_model_name" -T "$test_arff" -p 0 -distribution > "$pred_file" || exit 1
	fi
    
	echo "finished evaluate SVM and write predictions"

	# produce ARFF file in submission format
	pred_arff=$eval_dir/$feat_name.SMO.C$C.L$L.arff
	if [ ! -f "$pred_arff" ]; then
		perl format_pred.pl $test_arff $pred_file $pred_arff $lab || exit 1
	fi

	echo "Created submission format ARFF: $pred_arff"

	ref_arff=$feat_dir/$feat_name.devel.arff
	if [ -f "$ref_arff" ]; then
		echo "Found reference ARFF: $ref_arff"
		result_file=$eval_dir/`basename $pred_file .pred`.result
		if [ ! -f $result_file ]; then
			perl score.pl $ref_arff $pred_arff $lab | tee $result_file
		else
			cat $result_file
		fi
	fi
#done
