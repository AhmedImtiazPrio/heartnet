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
