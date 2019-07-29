%%Extract cardiac segments from heart sound signals
for folder_idx=0:5
clearvars -except folder_idx
clc
%% Initialize Parameters
% folder_idx=4; %index for training folder [0 to 5]
max_audio_length=60;    %seconds
N=60;                   %order of filters
sr=1000;                %resampling rate
nsamp = 2500;           %number of samples in each cardiac cycle segment
X=[];
Y=[];
file_name=[];
states=[];

%% Initialize paths

datapath=['../data/training/training-' 'a'+folder_idx '/'];
labelpath=['../data/training/' 'training-' 'a'+folder_idx '/REFERENCE-SQI.csv'];
savedir='../potes_1DCNN/';
exclude_text='../data/training/Recordings need to be removed in training-e.txt';

addpath(genpath('../cristhian.potes-204/'));
d=dir([datapath,'*.wav']);
num_files=size(d,1);

%% Initialize filter bank
% 
% Wn = 45*2/sr; % lowpass cutoff
% b1 = fir1(N,Wn,'low',hamming(N+1));
% Wn = [45*2/sr, 80*2/sr]; %bandpass cutoff
% b2 = fir1(N,Wn,hamming(N+1));
% Wn = [80*2/sr, 200*2/sr]; %bandpass cutoff
% b3 = fir1(N,Wn,hamming(N+1));
% Wn = 200*2/sr; %highpass cutoff
% b4 = fir1(N,Wn,'high',hamming(N+1));

%% Feature extraction using Springers Segmentation

load('Springer_B_matrix.mat');
load('Springer_pi_vector.mat');
load('Springer_total_obs_distribution.mat');
springer_options   = default_Springer_HSMM_options;
springer_options.use_mex = 1;

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
               
    [idx_states , last_idx]=get_states_python(assigned_states); %idx_states ncc x 4 matrix 
                                % containing starting index of segments 
    
%% Dividing signals into filter banks
    clear PCG
    PCG = PCG_resampled;

    nfb = 4;
    ncc = size(idx_states,1);
    x = nan(ncc,nsamp);
    for row=1:ncc
%         for fb=1:nfb
            if row == ncc % for the last complete cardiac cycle
                tmp = PCG(idx_states(row,1):last_idx-1);
            else
                tmp = PCG(idx_states(row,1):idx_states(row+1,1)-1);
            end
            N = nsamp-length(tmp); % append zeros at the end of cardiac cycle
            x(row,:) = [tmp; zeros(N,1)];
%         end
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

% %% Save Data
    sname=[savedir 'training-' 'a'+folder_idx '_noFIR' '.mat'];
    disp(['Saving ' sname])
    save(sname, 'X', 'Y', 'states', 'file_name');
 end
