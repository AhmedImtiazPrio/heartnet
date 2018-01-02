%% Spectral characteristics average per dataset
% figure('units','normalized','outerposition',[0 0 1 1]) %% for loglog plot maximize figure
a=[];
for i=0:5
clearvars -except i a
folder_idx=i; %index for training folder [0 to 5]
datapath=['/media/taufiq/Data/heart_sound/Heart_Sound/Physionet/training/training-' 'a'+folder_idx '/'];
exclude_text='/media/taufiq/Data/heart_sound/Heart_Sound/Physionet/2016-07-25_Updated files for Challenge 2016/Recordings need to be removed in training-e.txt';
labelpath=['/media/taufiq/Data/heart_sound/Heart_Sound/Physionet/2016-07-25_Updated files for Challenge 2016/20160725_Reference with signal quality results for training set/' 'training-' 'a'+folder_idx '/REFERENCE_withSQI.csv'];

d=dir([datapath,'*.wav']);
num_files=size(d,1);

window = 1; %   in seconds
overlap = .5 ; %  fraction overlap during STFT
nFFT = 120*2000;
%% Import list of files to be excluded

exclude = importlist(exclude_text);
ftype = repmat('.wav',[length(exclude),1]); 
exclude = strcat(exclude,ftype); % list of files to be excluded from training-e

%% Importing labels
labels=importlabel(labelpath); % first column normal(-1)/abnormal(1) second column good(1)/bad(0)
label_pointer=1; % label saving index

Avg = [];
for file_idx=1:num_files
%% Importing signals
    if folder_idx==4    % if dataset is training-e
        if sum(cell2mat(strfind(exclude,d(file_idx).name))) % if file is found in exclude list
            continue;
        end
    end
    
    if labels(file_idx)==1
        continue;
    end
    
    fname=[datapath,d(file_idx).name];
    [PCG,Fs] = audioread(fname);

    
    
%% Calculate Spectrogram

    [S,F,T] = spectrogram(PCG,window*Fs,overlap*window*Fs,nFFT,Fs);
    
    Avg = [Avg; mean(abs(real(S')))];
    
    
    
end
%% Surf Plot
% 
% figure('units','normalized','outerposition',[0 0 1 1]) %% maximize figure
% figure_path='/media/taufiq/Data/heart_sound/Presentation/Presentation 2/';
% [X,Y] = meshgrid(F,[1:size(Avg,1)]);
% s=surf(X,Y,Avg,mag2db(Avg))
% s.EdgeColor='flat';
% s.FaceColor='interp';
% shading interp
% view(30,40)
% colormap jet
% colorbar
% 
% set(gca,'xscale','log')
% set(gca,'zscale','log')
% xlim([F(1) F(end)])
% ylim([1 size(Avg,1)])
% zlim([10^-4 1000])
% grid on
% xlabel('Frequency (Hz)');
% ylabel('Recording #');
% zlabel('Magnitude');
% title(['training-' 'a'+folder_idx ' ' 'normal']);
% hC = colorbar
% hC.Label.String = 'dB'
% savefig([figure_path 'training-' 'a'+folder_idx 'normalsurf.fig'])
% print([figure_path 'training-' 'a'+folder_idx 'normalsurf'],'-dpng')

%% LogLog Plot

% plot(F,mean(Avg))
% hold on
% end
% %%
% set(gca,'xscale','log')
% set(gca,'yscale','log')
% xlim([F(1) F(end)])
% ylim([10^-4 1000])
% grid on
% xlabel('Frequency (Hz)');
% ylabel('Magnitude (dB)');
% title(['Freq characteristics per sensor (normal)']);
a=[a;mean(Avg)];
end