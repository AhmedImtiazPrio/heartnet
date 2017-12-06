clear
clc
rng('default')
fold = 3; % number of normal/abnormals in new folds
datapath='/media/taufiq/Data/heart_sound/feature/potes_1DCNN/';
fold_path='/media/taufiq/Data/heart_sound/feature/potes_1DCNN/balancedCV/';
fold_save='/media/taufiq/Data/heart_sound/feature/potes_1DCNN/balancedCV/folds/'

%% Accumulating all the data together / Loading data

if ~(exist([datapath 'data_all.mat'], 'file') == 2)
                               % checks if data has been accumulated before
    data=[];
    labels=[];
    cc_idx=[];      %cardiac cycle indices
    filenames=[];   %filenames of each segment

    for folder_idx=0:5
    loadfile=[datapath 'training-' 'a'+folder_idx '.mat'];
    load(loadfile);
    data=[data;X];
    labels=[labels;Y];
    cc_idx=[cc_idx;states];
    filenames=[filenames;file_name];
    end
    save([datapath 'data_all.mat'],'data','labels','cc_idx','filenames')
else
    load([datapath 'data_all.mat'])
end

%% Start creating folds
for i=0:fold,
    trainX=[];
    trainY=[];
    valX=[];
    valY=[];
    
    %% Partition training data
    loadfile=[fold_path 'train' num2str(i) '.txt'];
    train=sort(strcat(importlist(loadfile),'.wav'));
    for idx=1:length(train)
        trainX = [trainX;data(filenames==train(idx),:,:)];
        trainY = [trainY;labels(filenames==train(idx),:)]; 
    end
    %% Partition validation data
    loadfile=[fold_path 'validation' num2str(i) '.txt'];
    val=sort(strcat(importlist(loadfile),'.wav'));
    for idx=1:length(val)
        valX = [valX;data(filenames==val(idx),:,:)];
        valY = [valY;labels(filenames==val(idx),:)]; 
    end  
    
    save_name = ['fold' num2str(i) '.mat'];
    disp(['saving' ' ' save_name])
    save([fold_save save_name], 'trainX', 'trainY', 'valX', 'valY');
end
