# Towards Stethoscope Invariant Heart Sound Abnormality Detection using Learnable Filterbanks
### Submitted to *IEEE TBME*

**Objective:** Cardiac auscultation is the most practiced non-invasive and cost-effective procedure for the early diagnosis
of heart diseases. While machine learning based systems can aid in automatically screening patients, the robustness of these systems is affected by the phonocardiogram (PCG) acquisition device, i.e., the stethoscope. This paper studies the adverse effect of stethoscope/sensor variability on PCG signal classification and develops strategies to address this problem.

**Methods:** We propose a novel Convolutional Neural Network (CNN) layer, consisting of time-convolutional (tConv) units, that emulate Finite Impulse Response (FIR) filters. These filter coefficients can be updated via backpropagation and be stacked in the front-end of the network as a learnable filterbank. The filters can incorporate properties such as linear/zero phase-response and symmetry while ensuring robustness towards stethoscope variations by attenuating sensor dependent patterns.

**Results:** Our methods are evaluated using multi-source heart sound recordings obtained from the 2016 PhysioNet/CinC Challenge Dataset and the 2018 INTERSPEECH ComParE Heart Beats Sub-Challenge Dataset. The proposed learnable filterbank CNN architecture surpasses the top-scoring methods from both of these challenges on our multi-domain evaluation tasks. Our systems achieved relative improvements of up to 11.84% in terms of modified accuracy (Macc), compared to state-of-the-art methods.

**Conclusion:** The results demonstrate the effectiveness the proposed learnable filterbank CNN architecture in achieving robustness towards sensor variations in PCG signals.

**Significance:** To the best of our knowledge, this is the first research work that addresses the domain variability challenge for heart sound classification.

## Requirements
* Python 3.6.3
* Matlab 2017b
* Keras 2.2.4
* Tensorflow 1.12.0
* Sklearn 0.19.1
* Tensorboard

## How To Run
#### Data Preparation:
 First download the *data* folder from this [GoogleDrive Link](https://drive.google.com/open?id=1MPBhemO6XeDfjIm5-SOQUGvmzIl0Hx03)<br />
Place [Physionet dataset](https://physionet.org/content/challenge-2016/1.0.0/#files) (not included in the provided *data* folder) in the corresponding folders inside the *data/physionet/training* folder.
The csv files containing the labels should be put inside the corresponding folders inside the *labels* folder and all of them should have the same name, currently 'REFERENCE_withSQI.csv'. 
If you change the name you'll have to rename the variable *labelpath* in  *extract_segments.m* and *extract_segments_noFIR.m*<br /> 
Run *extract_segments_noFIR.m* it first then run *data_fold_noFIR.m* to create data fold in *mat* format which will be loaded by the model for training and testing.
*fold0_noFIR.mat* is given inside *data/feature/folds* for convenience, so that you don't have to download the whole physionet dataset and extract data for training and testing.

#### Training:
For Training run the *trainer.py* and provide a dataset name (or fold name) i.e. *fold0_noFIR*. The command should be like this : 
~~~~{.python}
python trainer.py fold0_noFIR
~~~~
Other parameters like epochs, verbose, batch_size, pre-trained model path can be passed as arguments. 
 ~~~~{.python}
python trainer.py fold0_noFIR --epochs 300 --batch_size 1000 
~~~~


#### Re-Generate Results:
Run the *heartnet testbench.ipynb* on Jupyter Notebook from the beginning until the block named *Model.Predict* . 
Select a *log_name* by uncommenting one from the **LOG name** block. 
The trained models for *"heartnet type2 tconv"* and *"potes algorithm"* is given in the *log* and *model* directory. 
These models are trained on *fold0_noFIR* which is included in the data folder.  
To do the McNemer test read the instruction given in the **LOG name** block of the notebook.
To plot roc curve run the **ROC curve** block.
