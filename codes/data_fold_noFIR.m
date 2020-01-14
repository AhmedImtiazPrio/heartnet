clear
clc
rng('default')
fold = 3; % number of normal/abnormals in new folds
datapath='../data/feature/';
fold_text_path='../data/feature/folds/text/';
fold_save='../data/feature/folds/';

%% Accumulating all the data together / Loading data

if ~(exist([datapath 'data_all_noFIR.mat'], 'file') == 2)
                               % checks if data has been accumulated before
    data=[];
    labels=[];
    cc_idx=[];      %cardiac cycle indices
    filenames=[];   %filenames of each segment
    val_parts=[];   %number of cc per recording
    train_parts=[];  %number of cc per recording
    
    for folder_idx=0:5
    loadfile=[datapath 'training-' 'a'+folder_idx '_noFIR' '.mat'];
    load(loadfile);
    data=[data;X];
    labels=[labels;Y];
    cc_idx=[cc_idx;states];
    filenames=[filenames;file_name];
    end
    save([datapath 'data_all_noFIR.mat'],'data','labels','cc_idx','filenames')
else
    load([datapath 'data_all_noFIR.mat'])
end

%% Start creating folds
for i=0:fold
    addpath(genpath('matlabUtils/'));
    trainX=[];
    trainY=[];
    valX=[];
    valY=[];
    filestrain=[];
    filesval=[];
    cc_train=[];
    cc_val=[];
    train_parts=[];
    val_parts=[];
    
    %% Partition training data
    cwd = pwd;
    loadfile=[cwd(1:end-5) fold_text_path(3:end) 'train' num2str(i) '.txt']; % creating full path to avoid file exception
    train=sort(strcat(importlist(loadfile),'.wav'));
    for idx=1:length(train)
        trainX = [trainX;data(filenames==train(idx),:,:)];
        trainY = [trainY;labels(filenames==train(idx),:)];
        filestrain = [filestrain; filenames(filenames==train(idx))];
          cc_train  = [cc_train;cc_idx(filenames==train(idx))];
          train_parts = [train_parts;sum(filenames==train(idx))];
    end
    %% Partition validation data
    cwd = pwd;
    loadfile=[cwd(1:end-5) fold_text_path(3:end) 'validation' num2str(i) '.txt'];
    val=sort(strcat(importlist(loadfile),'.wav'));
    for idx=1:length(val)
        valX = [valX;data(filenames==val(idx),:,:)];
        valY = [valY;labels(filenames==val(idx),:)]; 
        filesval = [filesval; filenames(filenames==val(idx))];
        cc_val  = [cc_val;cc_idx(filenames==val(idx))];
        val_parts = [val_parts;sum(filenames==val(idx))];
    end  
    
    save_name = ['fold' num2str(i) '_noFIR' '.mat'];
    disp(['saving' ' ' save_name])
%     clearvars -except cc_train train_parts cc_val val_parts
    save([fold_save save_name], 'trainX', 'trainY', 'valX', 'valY', 'cc_train',...
            'train_parts', 'cc_val', 'val_parts');
end
