from __future__ import absolute_import, division, print_function
from scipy.io import loadmat
import arff
import numpy as np
import os

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVC


FBANK_feat = os.path.join('FBANK','binnedFeat.mat')
FBANK_label = os.path.join('FBANK','binnedLabels.csv')
ComParE_feat = os.path.join('openSMILEall_PCG.mat')
BOAW_feat = os.path.join('baseline_boaw','feat','boawFeat.4096.arff')


#### Train SVC on FBANK Features ####



#### Train SVC on ComParE Features ####



#### Train SVC on BOAW Features ####




