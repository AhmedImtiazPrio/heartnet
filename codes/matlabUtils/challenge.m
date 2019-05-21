function classifyResult = challenge(recordName)
%
% Sample entry for the 2016 PhysioNet/CinC Challenge.
%
% INPUTS:
% recordName: string specifying the record name to process
%
% OUTPUTS:
% classifyResult: integer value where
%                     1 = abnormal recording
%                    -1 = normal recording
%                     0 = unsure (too noisy)
%
% To run your entry on the entire training set in a format that is
% compatible with PhysioNet's scoring enviroment, run the script
% generateValidationSet.m
%
% The challenge function requires that you have downloaded the challenge
% data 'training_set' in a subdirectory of the current directory.
%    http://physionet.org/physiobank/database/challenge/2016/
%
% This dataset is used by the generateValidationSet.m script to create
% the annotations on your training set that will be used to verify that
% your entry works properly in the PhysioNet testing environment.
%
%
% Version 1.0
%
%
% Written by: Chengyu Liu, Fubruary 21 2016
%             chengyu.liu@emory.edu
%
% Last modified by:
%
%

%% Load the trained parameter matrices for Springer's HSMM model.
% The parameters were trained using 409 heart sounds from MIT heart
% sound database, i.e., recordings a0001-a0409.
load('Springer_B_matrix.mat');
load('Springer_pi_vector.mat');
load('Springer_total_obs_distribution.mat');
load('parms_cnn.mat');
load('learned_parms.mat')
parms.maxpooling = 2;

N=60; sr = 1000; 
Wn = 45*2/sr; 
b1 = fir1(N,Wn,'low',hamming(N+1));
Wn = [45*2/sr, 80*2/sr];
b2 = fir1(N,Wn,hamming(N+1));
Wn = [80*2/sr, 200*2/sr];
b3 = fir1(N,Wn,hamming(N+1));
Wn = 200*2/sr;
b4 = fir1(N,Wn,'high',hamming(N+1));

%% Load data and resample data
springer_options   = default_Springer_HSMM_options;
springer_options.use_mex = 1;
%[PCG, Fs1, nbits1] = wavread([recordName '.wav']);  % load data
[PCG,Fs1] = audioread([recordName '.wav']);  % load data

if length(PCG)>60*Fs1
    PCG = PCG(1:60*Fs1);
end
            
% resample to 1000 Hz
PCG_resampled = resample(PCG,springer_options.audio_Fs,Fs1); % resample to springer_options.audio_Fs (1000 Hz)
% filter the signal between 25 to 400 Hz
PCG_resampled = butterworth_low_pass_filter(PCG_resampled,2,400,springer_options.audio_Fs, false);
PCG_resampled = butterworth_high_pass_filter(PCG_resampled,2,25,springer_options.audio_Fs);
% remove spikes
PCG_resampled = schmidt_spike_removal(PCG_resampled,springer_options.audio_Fs);

%% Running runSpringerSegmentationAlgorithm.m to obtain the assigned_states
assigned_states = runSpringerSegmentationAlgorithm(PCG_resampled,... 
                springer_options.audio_Fs,... 
                Springer_B_matrix, Springer_pi_vector,...
                Springer_total_obs_distribution, false);

% get states
idx_states = get_states(assigned_states);

features_time = get_features_time(PCG_resampled,idx_states);
features_freq = get_features_frequency(PCG_resampled,idx_states);
features = [features_time, features_freq];

% mu = repmat(mu,size(features,1),1);
% sigma1 = repmat(sigma1,size(features,1),1);
features = (features-mu1)./sigma1;

% classification using boosting
prb_boosting = boostedHII_predict(features,cvres_full.clf_full,'probability');

% filter signal in 4 different frequency bands,
% [0,45],[45-80],[80-200],[200-400]
clear PCG
PCG(:,1) = filtfilt(b1,1,PCG_resampled);
PCG(:,2) = filtfilt(b2,1,PCG_resampled);
PCG(:,3) = filtfilt(b3,1,PCG_resampled);
PCG(:,4) = filtfilt(b4,1,PCG_resampled);

nfb = 4;
nsamp = 2500;
ncc = size(idx_states,1)-1;
X = nan(ncc,nsamp,nfb);
for row=1:ncc
    for fb=1:nfb
        tmp = PCG(idx_states(row,1):idx_states(row+1,1),fb);
        N = nsamp-length(tmp);
        % append zeros at the end of cardiac cycle
        X(row,:,fb) = [tmp; zeros(N,1)];
    end
end

% run cnn feedforward
res = nan(size(X,1),2);
for sample = 1:size(X,1)  
    s = squeeze(X(sample,:,:));
    res(sample,:) = feed_forward_cnn(s,parms);
end
prb_cnn = mean(res(:,2));

if ((prb_cnn > 0.4) || (prb_boosting > 0.4))
    classifyResult = 1;
else
    classifyResult = -1;
end


end

%%
function idx_states = get_states(assigned_states)
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

    indx2                = indx(K:end);
    rem                  = mod(length(indx2),4);
    indx2(end-rem+1:end) = [];
    idx_states           = reshape(indx2,4,length(indx2)/4)';
end

%% 
function features = get_features_time(PCG,idx_states)
    %% Feature calculation
    m_RR        = round(mean(diff(idx_states(:,1))));             % mean value of RR intervals
    sd_RR       = round(std(diff(idx_states(:,1))));              % standard deviation (SD) value of RR intervals
    mean_IntS1  = round(mean(idx_states(:,2)-idx_states(:,1)));            % mean value of S1 intervals
    sd_IntS1    = round(std(idx_states(:,2)-idx_states(:,1)));             % SD value of S1 intervals
    mean_IntS2  = round(mean(idx_states(:,4)-idx_states(:,3)));            % mean value of S2 intervals
    sd_IntS2    = round(std(idx_states(:,4)-idx_states(:,3)));             % SD value of S2 intervals
    mean_IntSys = round(mean(idx_states(:,3)-idx_states(:,2)));            % mean value of systole intervals
    sd_IntSys   = round(std(idx_states(:,3)-idx_states(:,2)));             % SD value of systole intervals
    mean_IntDia = round(mean(idx_states(2:end,1)-idx_states(1:end-1,4)));  % mean value of diastole intervals
    sd_IntDia   = round(std(idx_states(2:end,1)-idx_states(1:end-1,4)));   % SD value of diastole intervals

    for i=1:size(idx_states,1)-1
        R_SysRR(i)  = (idx_states(i,3)-idx_states(i,2))/(idx_states(i+1,1)-idx_states(i,1))*100;
        R_DiaRR(i)  = (idx_states(i+1,1)-idx_states(i,4))/(idx_states(i+1,1)-idx_states(i,1))*100;
        R_SysDia(i) = R_SysRR(i)/R_DiaRR(i)*100;
        
        %skewness
        SK_S1(i)  = skewness(PCG(idx_states(i,1):idx_states(i,2)));
        SK_Sys(i) = skewness(PCG(idx_states(i,2):idx_states(i,3)));
        SK_S2(i)  = skewness(PCG(idx_states(i,3):idx_states(i,4)));
        SK_Dia(i) = skewness(PCG(idx_states(i,4):idx_states(i+1,1)));
        
        % kurtosis
        KU_S1(i)  = kurtosis(PCG(idx_states(i,1):idx_states(i,2)));
        KU_Sys(i) = kurtosis(PCG(idx_states(i,2):idx_states(i,3)));
        KU_S2(i)  = kurtosis(PCG(idx_states(i,3):idx_states(i,4)));
        KU_Dia(i) = kurtosis(PCG(idx_states(i,4):idx_states(i+1,1)));

        P_S1(i)     = sum(abs(PCG(idx_states(i,1):idx_states(i,2))))/(idx_states(i,2)-idx_states(i,1));
        P_Sys(i)    = sum(abs(PCG(idx_states(i,2):idx_states(i,3))))/(idx_states(i,3)-idx_states(i,2));
        P_S2(i)     = sum(abs(PCG(idx_states(i,3):idx_states(i,4))))/(idx_states(i,4)-idx_states(i,3));
        P_Dia(i)    = sum(abs(PCG(idx_states(i,4):idx_states(i+1,1))))/(idx_states(i+1,1)-idx_states(i,4));
        if P_S1(i)>0
            P_SysS1(i) = P_Sys(i)/P_S1(i)*100;
        else
            P_SysS1(i) = 0;
        end
        if P_S2(i)>0
            P_DiaS2(i) = P_Dia(i)/P_S2(i)*100;
        else
            P_DiaS2(i) = 0;
        end
    end

    m_Ratio_SysRR   = mean(R_SysRR);  % mean value of the interval ratios between systole and RR in each heart beat
    sd_Ratio_SysRR  = std(R_SysRR);   % SD value of the interval ratios between systole and RR in each heart beat
    m_Ratio_DiaRR   = mean(R_DiaRR);  % mean value of the interval ratios between diastole and RR in each heart beat
    sd_Ratio_DiaRR  = std(R_DiaRR);   % SD value of the interval ratios between diastole and RR in each heart beat
    m_Ratio_SysDia  = mean(R_SysDia); % mean value of the interval ratios between systole and diastole in each heart beat
    sd_Ratio_SysDia = std(R_SysDia);  % SD value of the interval ratios between systole and diastole in each heart beat
    
    mSK_S1 = mean(SK_S1);
    sdSK_S1 = std(SK_S1);
    mSK_Sys = mean(SK_Sys);
    sdSK_Sys = std(SK_Sys);
    mSK_S2 = mean(SK_S2);
    sdSK_S2 = std(SK_S2);
    mSK_Dia = mean(SK_Dia);
    sdSK_Dia = std(SK_Dia);
    
    mKU_S1 = mean(KU_S1);
    sdKU_S1 = std(KU_S1);
    mKU_Sys = mean(KU_Sys);
    sdKU_Sys = std(KU_Sys);
    mKU_S2 = mean(KU_S2);
    sdKU_S2 = std(KU_S2);
    mKU_Dia = mean(KU_Dia);
    sdKU_Dia = std(KU_Dia);

    indx_sys = find(P_SysS1>0 & P_SysS1<100);   % avoid the flat line signal
    if length(indx_sys)>1
        m_Amp_SysS1  = mean(P_SysS1(indx_sys)); % mean value of the mean absolute amplitude ratios between systole period and S1 period in each heart beat
        sd_Amp_SysS1 = std(P_SysS1(indx_sys));  % SD value of the mean absolute amplitude ratios between systole period and S1 period in each heart beat
    else
        m_Amp_SysS1  = 0;
        sd_Amp_SysS1 = 0;
    end
    indx_dia = find(P_DiaS2>0 & P_DiaS2<100);
    if length(indx_dia)>1
        m_Amp_DiaS2  = mean(P_DiaS2(indx_dia)); % mean value of the mean absolute amplitude ratios between diastole period and S2 period in each heart beat
        sd_Amp_DiaS2 = std(P_DiaS2(indx_dia));  % SD value of the mean absolute amplitude ratios between diastole period and S2 period in each heart beat
    else
        m_Amp_DiaS2  = 0;
        sd_Amp_DiaS2 = 0;
    end

    features = [m_RR, sd_RR, mean_IntS1, sd_IntS1, mean_IntS2, sd_IntS2, mean_IntSys, sd_IntSys, mean_IntDia, ...
               sd_IntDia, m_Ratio_SysRR, sd_Ratio_SysRR, m_Ratio_DiaRR, sd_Ratio_DiaRR, m_Ratio_SysDia, ...
               sd_Ratio_SysDia, m_Amp_SysS1, sd_Amp_SysS1, m_Amp_DiaS2, sd_Amp_DiaS2,mSK_S1,sdSK_S1,...
               mSK_Sys,sdSK_Sys,mSK_S2,sdSK_S2,mSK_Dia,sdSK_Dia, mKU_S1,sdKU_S1,...
               mKU_Sys,sdKU_Sys,mKU_S2,sdKU_S2,mKU_Dia,sdKU_Dia];
end
%%
%%
function features = get_features_frequency(PCG,idx_states)
    NFFT = 256;
    f = (0:NFFT/2-1)/(NFFT/2)*500;
    freq_range = [25,45;45,65;65,85;85,105;105,125;125,150;150,200;200,300;300,500];
    p_S1  = nan(size(idx_states,1)-1,NFFT/2);
    p_Sys = nan(size(idx_states,1)-1,NFFT/2);
    p_S2  = nan(size(idx_states,1)-1,NFFT/2);
    p_Dia = nan(size(idx_states,1)-1,NFFT/2);
    for row=1:size(idx_states,1)-1
        s1 = PCG(idx_states(row,1):idx_states(row,2));
        s1 = s1.*hamming(length(s1));
        Ft = fft(s1,NFFT);
        p_S1(row,:) = abs(Ft(1:NFFT/2));
        
        sys = PCG(idx_states(row,2):idx_states(row,3));
        sys = sys.*hamming(length(sys));
        Ft  = fft(sys,NFFT);
        p_Sys(row,:) = abs(Ft(1:NFFT/2));
        
        s2 = PCG(idx_states(row,3):idx_states(row,4));
        s2 = s2.*hamming(length(s2));
        Ft = fft(s2,NFFT);
        p_S2(row,:) = abs(Ft(1:NFFT/2));
        
        dia = PCG(idx_states(row,4):idx_states(row+1,1));
        dia = dia.*hamming(length(dia));
        Ft  = fft(dia,NFFT);
        p_Dia(row,:) = abs(Ft(1:NFFT/2));
    end
    P_S1 = nan(1,size(freq_range,1));
    P_Sys = nan(1,size(freq_range,1));
    P_S2 = nan(1,size(freq_range,1));
    P_Dia = nan(1,size(freq_range,1));
    for bin=1:size(freq_range,1)
        idx = (f>=freq_range(bin,1)) & (f<freq_range(bin,2));
        P_S1(1,bin) = median(median(p_S1(:,idx)));
        P_Sys(1,bin) = median(median(p_Sys(:,idx)));
        P_S2(1,bin) = median(median(p_S2(:,idx)));
        P_Dia(1,bin) = median(median(p_Dia(:,idx)));
    end
    features = [P_S1, P_Sys, P_S2, P_Dia];
end

function y = boostedHII_predict(X,clf,output_type,max_round)
% FUNCTION y = boostedHII_predict(X,clf,output_type)
%   *** INPUTS ***
%   X:
%   clf:
%   output_type:
%
%   *** OUTPUTS ***
%   y:

    if ~exist('output_type','var'); output_type = ''; end;
    if ~exist('max_round','var'); max_round = []; end;

    if ~numel(max_round); max_round = inf; end;

    weak_learners = clf.weak_learners;
    stacking = clf.stacking;

    [n,p] = size(X);
    X = [X,ones(n,1)];
    p = p + 1;

    if strcmp(clf.missingDataHandler.method,'abstain')
        % Then we have nothing to do...
    elseif strcmp(clf.missingDataHandler.method,'mean')
        % Use the imputed values
        for j=1:p
            X(isnan(X(:,j)),j) = clf.missingDataHandler.imputedVals(j);
        end
    elseif strcmp(clf.missingDataHandler.method,'multivariate_normal_model')
        % Then use the multivariate normal to impute values on each trial
        % First, get all of the distinct missing patterns
        [MP,~,pattern_inds] = unique(~isnan(X(:,1:end-1)),'rows');
        [pattern_index,pattern_inds] = sort(pattern_inds,'ascend');
        j = 0;
        while j < n
            fprintf('%d\n',j);pause(1e-5);
            j = j + 1;
            jStart = j;
            index = pattern_index(j);
            while j <= n && pattern_index(j) == index
                j = j + 1;
            end
            j = j - 1;

            inds = pattern_inds(jStart:j);

            exists_locs = find(MP(index,:));
            missing_locs = setdiff(1:p-1,exists_locs);

            tmpCov = clf.missingDataHandler.covMat;
            tmpCov = tmpCov([missing_locs,exists_locs],:);
            tmpCov = tmpCov(:,[missing_locs,exists_locs]);

            S12 = tmpCov(1:numel(missing_locs),(numel(missing_locs)+1):end);
            S22 = tmpCov((numel(missing_locs)+1):end,(numel(missing_locs)+1):end);

            impute_vals = repmat(clf.missingDataHandler.means(missing_locs),1,numel(inds)) + S12*(S22\(X(inds,exists_locs)' - repmat(clf.missingDataHandler.means(exists_locs),1,numel(inds))));

            X(inds,missing_locs) = impute_vals';
        end  
    else
        error('Unknown missingDataHandler method');
    end

    XM = double(~isnan(X));
    yfull = zeros(n,p);yfullind=0;%min(max_round,numel(decision_stumps)));yfullind = 0;
    y = zeros(n,1);
    for j=1:p
        if ~numel(weak_learners{j}); continue; end;

        locs = find(XM(:,j));
        ycurr = zeros(n,1);
        for k=1:numel(weak_learners{j})
            ds = weak_learners{j}{k};
            if ds.boosting_round > max_round
                break;
            end
            if isfield(ds,'bias')
                ycurr(locs) = ycurr(locs) + ds.bias;
            end

            switch ds.type
                case 'constant'
                    % There's nothing left to do
                case 'logistic'
                    ycurr(locs) = ycurr(locs) + ds.alpha*(2./(1+exp(-(ds.a*X(locs,j)+ds.b)))-1);
                case 'stump'                
                    % First assume it's positive and then double correct the negatives
                    ycurr(locs) = ycurr(locs) + ds.alpha;
                    neg_locs = find(X(locs,j) < ds.threshold);
                    ycurr(locs(neg_locs)) = ycurr(locs(neg_locs)) - 2*ds.alpha;  
                otherwise
                    error('Unknown weak learner type');
            end
        end

        if isfield(stacking,'mf_coeffs')
            c = stacking.mf_coeffs{j}(1) + XM*stacking.mf_coeffs{j}(2:end)';
            ycurr = ycurr.*c;
        end

        yfullind = yfullind + 1;
        yfull(:,yfullind) = ycurr;

        y = y + ycurr;
    end

    if strcmp(output_type,'')
        % Then don't do anything
    elseif strcmp(output_type,'probability')
        y = 1./(1+exp(-y));
    else
        error('Unknown output_type');
    end
end
    
