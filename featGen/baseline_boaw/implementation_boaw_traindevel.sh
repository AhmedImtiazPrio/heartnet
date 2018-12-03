#!/bin/sh


# path to your feature directory (htk files)
feat_dir=../mfcc

# openXBOW parameters
openXBOW_arff_labels="0,1,2"
openXBOW_attributes_string="nt1[65]2[65]c"
openXBOW_num_assignments=10
lab=$(echo $openXBOW_size_codebook | sed 's/,/+/g')
lab=$(echo "$lab +2" | bc)  # Must be the sum of codebook sizes + 2

# Compute BoAW
if [ ! -s "$train_boaw_arff" ]; then
	java -Xmx$jvm_mem -jar openXBOW.jar -writeName -i "$train_lld_arff" -o "$train_boaw_arff" -standardizeInput -log -size "$openXBOW_size_codebook" -B "$model_dir/codebook" -a "$openXBOW_num_assignments" -attributes "$openXBOW_attributes_string" -arffLabels "$openXBOW_arff_labels"
fi
if [ ! -s "$test_boaw_arff" ]; then
	java -Xmx$jvm_mem -jar openXBOW.jar -writeName -i "$test_lld_arff" -o "$test_boaw_arff" -b "$model_dir/codebook" -attributes "$openXBOW_attributes_string" -arffLabels "$openXBOW_arff_labels"
fi


