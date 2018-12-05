from __future__ import absolute_import, division, print_function
from librosa import load, feature
import os
import numpy as np
import pandas as pd
from scipy.io import savemat

def partition(data,parts=10,overlap=.3):
    out= []
    size = len(data)
    window_size = int(np.floor(size/(1+(parts-1)*(1-overlap))))
    window_idx = [int(np.floor(part*(1-overlap)*window_size)) for part in range(parts)]
    for each in window_idx:
        out.append(data[each:each+window_size])
    return np.asarray(out)

wav_dir = os.path.join('..','wav')
list_wav = os.listdir(wav_dir)
n_mels = 41
parts = 10
overlap = .3
mfsc_hop_length = 512

dataX = []
filenames = []
for pcg in list_wav:
    print("Extracting Features from {}".format(pcg))
    data,sr = load(os.path.join(wav_dir,pcg))
    time = len(data)/sr
    data = partition(data,parts=parts,overlap=overlap)
    n_time = int(np.ceil(data.shape[1]/mfsc_hop_length))
    feats = []
    spect = np.zeros((parts,n_mels,n_time))
    delta = np.zeros((parts,n_mels,n_time-1))
    deltadelta = np.zeros((parts,n_mels,n_time-2))
    for part in range(data.shape[0]):
        spect[part,:,:] = feature.melspectrogram(data[part],sr=sr,power=1.,n_mels=n_mels,hop_length=mfsc_hop_length)
        delta[part,:,:] = np.transpose(np.asarray([spect[part,:,each]-spect[part,:,each+1] for each in range(spect.shape[-1]-1)]))
        deltadelta[part,:,:] = np.transpose(np.asarray([delta[part,:,each]-delta[part,:,each+1] for each in range(delta.shape[-1]-1)]))

        feats.append(np.mean(spect[part,:,:], axis=1))
        feats.append(np.std(spect[part,:,:], axis=1))
        feats.append(np.mean(delta[part,:,:], axis=1))
        feats.append(np.std(delta[part,:,:], axis=1))
        feats.append(np.mean(deltadelta[part,:,:], axis=1))
        feats.append(np.std(deltadelta[part,:,:], axis=1))

    feats.append(np.mean(np.hstack(spect),axis=1))
    feats.append(np.std(np.hstack(spect),axis=1))
    feats.append(np.mean(np.hstack(delta),axis=1))
    feats.append(np.std(np.hstack(delta),axis=1))
    feats.append(np.mean(np.hstack(deltadelta),axis=1))
    feats.append(np.std(np.hstack(deltadelta),axis=1))
    feats.append(time)

    dataX.append(np.hstack(feats))
    filenames.append(pcg)

dataX = np.asarray(dataX)
df = pd.DataFrame(filenames,columns={'file_name'})
df.set_index('file_name',inplace=True)
df = df.join(pd.read_csv(os.path.join('..','wav.tsv'),delimiter='\t').set_index('file_name'))
print("Data Tensor Shape {}".format(dataX.shape))

df.to_csv('binnedLabels.csv')
savemat('binnedFeat.mat',{'feats':dataX})
print("Saving Complete")







