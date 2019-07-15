clear
clc
rng('default')
fold = 3; % number of normal/abnormals in new folds
datapath='../potes_1DCNN/';
valid_path='../data/validation/RECORDS';
abnorpath='../data/training/';
exclude_text='../data/training/Recordings need to be removed in training-e.txt';
fold_path='../potes_1DCNN/balancedCV/textfile/'

%% Import physionet validation lsit
valid0 = importlist(valid_path);
%% Import all normal/abnormal list
abnorlist=[];
norlist=[];
exclude = importfile(exclude_text);
for folder_idx=0:5
    loadfile=[abnorpath 'training-' 'a'+folder_idx '/RECORDS-abnormal'];
    abnorlist=[abnorlist;importfile(loadfile)];
    loadfile=[abnorpath 'training-' 'a'+folder_idx '/RECORDS-normal'];
    norlist=[norlist;importfile(loadfile)];
end
%% Excluding the ECG from training-e
for idx=1:size(exclude,1)
    abnorlist(contains(abnorlist,exclude(idx)))=[];   % inserts "nothingness" where it matches the excluded
                                                      % filenames
    norlist(contains(norlist,exclude(idx)))=[];  
end
list_main=[norlist;abnorlist];
%% exclude validation set1 lists
for idx=1:size(valid0,1)
    norlist(contains(norlist,valid0(idx)))=[]; 
    abnorlist(contains(abnorlist,valid0(idx)))=[];
end
%% Create folds
num_samples=floor(length(abnorlist)/fold); %number of folds
abnorlist=datasample(abnorlist,length(abnorlist),'Replace',false); %randomize abnorlist
norlist=datasample(norlist,length(norlist),'Replace',false);    %randomize norlist
valid=[];
for it=1:fold
    [Y,I]=datasample(abnorlist,num_samples,'Replace',false); % Take random samples
    valid{it}=Y;                                             % Store names
    abnorlist(I)=[];                                         % Remove sampled Data1
    [Y,I]=datasample(norlist,num_samples,'Replace',false);
    valid{it}=[valid{it};Y];
    norlist(I)=[];
    fID=fopen([fold_path 'validation' '1'+it-1 '.txt'],'w'); % save validation filenames
                                                            % in text files
    fprintf(fID,'%s\n',valid{it}{:});
    fclose(fID);
end

fID=fopen([fold_path 'validation0' '.txt'],'w');     % save physionet validation filenames
fprintf(fID,'%s\n',valid0);
fclose(fID);

%% Create corresponding training sets

train0=list_main;
for idx=1:size(valid0,1)  % Create corresponding training set for valid0
    train0(contains(train0,valid0(idx)))=[]; 
end

fID=fopen([fold_path 'train0' '.txt'],'w');     % save physionet validation filenames
fprintf(fID,'%s\n',train0{:});
fclose(fID);

train=cell(3,1);
for it=1:fold
    train{it}=list_main;
    for  idx=1:size(valid{it},1)
        train{it}(contains(train{it},valid{it}(idx)))=[];
    end
    fID=fopen([fold_path 'train' '1'+it-1 '.txt'],'w'); 
    fprintf(fID,'%s\n',train{it}{:});
    fclose(fID);
end
