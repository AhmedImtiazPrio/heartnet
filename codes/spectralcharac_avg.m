%% Spectral characteristics average per dataset
clear
folder_idx=4; %index for training folder [0 to 5]
datapath=['/media/taufiq/Data/heart_sound/Heart_Sound/Physionet/training/training-' 'a'+folder_idx '/'];
exclude_text='/media/taufiq/Data/heart_sound/Heart_Sound/Physionet/2016-07-25_Updated files for Challenge 2016/Recordings need to be removed in training-e.txt';

d=dir([datapath,'*.wav']);
num_files=size(d,1);

window = 1; %   in seconds
overlap = .5 ; %  fraction overlap during STFT
nFFT = 120*2000;
%% Import list of files to be excluded

exclude = importlist(exclude_text);
ftype = repmat('.wav',[length(exclude),1]); 
exclude = strcat(exclude,ftype); % list of files to be excluded from training-e

Avg = [];

for file_idx=1:num_files
%% Importing signals
    if folder_idx==4    % if dataset is training-e
        if sum(cell2mat(strfind(exclude,d(file_idx).name))) % if file is found in exclude list
            continue;
        end
    end
    
    fname=[datapath,d(file_idx).name];
    [PCG,Fs] = audioread(fname);

%% Calculate Spectrogram

    [S,F,T] = spectrogram(PCG,window*Fs,overlap*window*Fs,nFFT,Fs);
    
    Avg = [Avg; mean(abs(real(S')))];

end
plot(F,mean(Avg))
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title(['training-' 'a'+folder_idx]);