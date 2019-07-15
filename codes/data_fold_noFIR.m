clear
clc
rng('default')
fold = 3; % number of normal/abnormals in new folds
datapath='../potes_1DCNN/';
fold_path='../potes_1DCNN/balancedCV/';
fold_save='../potes_1DCNN/balancedCV/folds/';

%% Accumulating all the data together / Loading data

if ~(exist([datapath 'data_all_gt.mat'], 'file') == 2)
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
    save([datapath 'data_all_gt.mat'],'data','labels','cc_idx','filenames')
else
    load([datapath 'data_all_gt.mat'])
end

%% Start creating folds
for i=0:fold,
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
    loadfile=[fold_path 'train' num2str(i) '.txt'];
    train=sort(strcat(importlist(loadfile),'.wav'));
    for idx=1:length(train)
        trainX = [trainX;data(filenames==train(idx),:,:)];
        trainY = [trainY;labels(filenames==train(idx),:)];
        
        filestrain = [filestrain; filenames(filenames==train(idx))];
        filestrainnew=convertStringsToChars(filestrain(:));  
        
        cc_train  = [cc_train;cc_idx(filenames==train(idx))];
          train_parts = [train_parts;sum(filenames==train(idx))];
    end
    %% Partition validation data
    loadfile=[fold_path 'validation' num2str(i) '.txt'];
    val=sort(strcat(importlist(loadfile),'.wav'));
    for idx=1:length(val)
        valX = [valX;data(filenames==val(idx),:,:)];
        valY = [valY;labels(filenames==val(idx),:)]; 
        
        filesval = [filesval; filenames(filenames==val(idx))];
        filesvalnew=convertStringsToChars(filesval(:));
        
        cc_val  = [cc_val;cc_idx(filenames==val(idx))];
        val_parts = [val_parts;sum(filenames==val(idx))];
    end  
    
    save_name = ['fold' num2str(i) '_noFIR' '.mat'];
    disp(['saving' ' ' save_name])
%     clearvars -except cc_train train_parts cc_val val_parts
    save([fold_save save_name], 'trainX', 'trainY', 'valX', 'valY', 'cc_train',...
            'train_parts', 'cc_val', 'val_parts', 'filestrainnew', 'filesvalnew');
end
