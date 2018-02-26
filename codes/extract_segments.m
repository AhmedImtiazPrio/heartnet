%%Extract cardiac segments from heart sound signals
clearvars -except folder_idx
clc
%% Initialize Parameters
folder_idx=4; %index for training folder [0 to 5]
max_audio_length=60;    %seconds
N=60;                   %order of filters
sr=1000;                %resampling rate
nsamp = 2500;           %number of samples in each cardiac cycle segment
X=[];
Y=[];
file_name=[];
states=[];

%% Initialize paths

datapath=['/media/taufiq/Data/heart_sound/Heart_Sound/Physionet/training/training-' 'a'+folder_idx '/'];
labelpath=['/media/taufiq/Data/heart_sound/Heart_Sound/Physionet/2016-07-25_Updated files for Challenge 2016/20160725_Reference with signal quality results for training set/' 'training-' 'a'+folder_idx '/REFERENCE_withSQI.csv'];
savedir='/media/taufiq/Data/heart_sound/feature/potes_1DCNN/';
exclude_text='/media/taufiq/Data/heart_sound/Heart_Sound/Physionet/2016-07-25_Updated files for Challenge 2016/Recordings need to be removed in training-e.txt';

addpath(genpath('/media/taufiq/Data/heart_sound/Heart_Sound/codes/cristhian.potes-204/'));
d=dir([datapath,'*.wav']);
num_files=size(d,1);

%% Initialize filter bank

Wn = 45*2/sr; % lowpass cutoff
b1 = fir1(N,Wn,'low',hamming(N+1));
Wn = [45*2/sr, 80*2/sr]; %bandpass cutoff
b2 = fir1(N,Wn,hamming(N+1));
Wn = [80*2/sr, 200*2/sr]; %bandpass cutoff
b3 = fir1(N,Wn,hamming(N+1));
Wn = 200*2/sr; %highpass cutoff
b4 = fir1(N,Wn,'high',hamming(N+1));

%% Feature extraction using Springers Segmentation

load('Springer_B_matrix.mat');
load('Springer_pi_vector.mat');
load('Springer_total_obs_distribution.mat');
springer_options   = default_Springer_HSMM_options;
springer_options.use_mex = 1;
% springer_options
%% Importing labels
labels=importlabel(labelpath); % first column normal(-1)/abnormal(1) second column good(1)/bad(0)
label_pointer=1; % label saving index
%% Import list of files to be excluded

exclude = importlist(exclude_text);
ftype = repmat('.wav',[length(exclude),1]); 
exclude = strcat(exclude,ftype); % list of files to be excluded from training-e

for file_idx=1:num_files
%% Importing signals
    if folder_idx==4    % if dataset is training-e
        if sum(cell2mat(strfind(exclude,d(file_idx).name))) % if file is found in exclude list
            continue;
        end
    end
    
    fname=[datapath,d(file_idx).name];
    [PCG,Fs1] = audioread(fname);
    if length(PCG)>max_audio_length*Fs1
        PCG = PCG(1:max_audio_length*Fs1); % Clip signals to max_audio_length seconds
    end
        
%% Pre-processing (resample + bandpass + spike removal)

    % resample to 1000 Hz
    PCG_resampled = resample(PCG,springer_options.audio_Fs,Fs1); 
    % filter the signal between 25 to 400 Hz
    PCG_resampled = butterworth_low_pass_filter(PCG_resampled,2,400,springer_options.audio_Fs, false);
    PCG_resampled = butterworth_high_pass_filter(PCG_resampled,2,25,springer_options.audio_Fs);
    % remove spikes
    PCG_resampled = schmidt_spike_removal(PCG_resampled,springer_options.audio_Fs);
    
%% Run springer's segmentation

    assigned_states = runSpringerSegmentationAlgorithm(PCG_resampled,... 
                    springer_options.audio_Fs,... 
                    Springer_B_matrix, Springer_pi_vector,...
                    Springer_total_obs_distribution, false);
               
    [idx_states , last_idx]=get_states(assigned_states); %idx_states ncc x 4 matrix 
                                % containing starting index of segments 
    
%% Dividing signals into filter banks
    clear PCG
    PCG(:,1) = filtfilt(b1,1,PCG_resampled);
    PCG(:,2) = filtfilt(b2,1,PCG_resampled);
    PCG(:,3) = filtfilt(b3,1,PCG_resampled);
    PCG(:,4) = filtfilt(b4,1,PCG_resampled);


    nfb = 4;
    ncc = size(idx_states,1);
    x = nan(ncc,nsamp,nfb);
    for row=1:ncc
        for fb=1:nfb
            if row == ncc % for the last complete cardiac cycle
                tmp = PCG(idx_states(row,1):last_idx-1,fb);
            else
                tmp = PCG(idx_states(row,1):idx_states(row+1,1)-1,fb);
            end
            N = nsamp-length(tmp); % append zeros at the end of cardiac cycle
            x(row,:,fb) = [tmp; zeros(N,1)];
        end
        file_name=[file_name;string(d(file_idx).name)]; % matrix containing the filename
                                                % of each cardiac cycle
        Y=[Y;labels(label_pointer,:)];               % Class labels for each cardiac cycle
    end
    X=[X;x]; % matrix containing all cardiac cycles
    states=[states;idx_states]; % matrix containing 
                                %index of states of each cardiac cycle
    label_pointer=label_pointer+1;  % point at label for the next recording
                                    % increasing with each loop
    
end

%% Save Data
    sname=[savedir 'training-' 'a'+folder_idx '.mat'];
    %save(sname, 'X', 'Y', 'states', 'file_name');

    
%% function to extract state index
function [idx_states,last_idx] = get_states(assigned_states)
    indx = find(abs(diff(assigned_states))>0); % find the locations with changed states

    if assigned_states(1)>0   % for some recordings, there are state zeros at the beginning of assigned_states
        switch assigned_states(1)
            case 4
                K=1;
            case 3
                K=2;
            case 2
                K=3;
            case 1
                K=4;
        end
    else
        switch assigned_states(indx(1)+1)
            case 4
                K=1;
            case 3
                K=2;
            case 2
                K=3;
            case 1
                K=0;
        end
        K=K+1;
    end

    indx2                = indx(K:end); % K controls the starting cycle
                                        % of the segment. Starting cycle
                                        % is always kept 1 through the 
                                        % switch cases (!)
                                        
    rem                  = mod(length(indx2),4);
    last_idx             = length(indx2)-rem+1;
    indx2(last_idx:end) = []; % clipping the partial segments in the end
    idx_states           = reshape(indx2,4,length(indx2)/4)'; % idx_states 
                            % reshaped into a no.segments X 4 sized matrix
                            % containing state indices
end