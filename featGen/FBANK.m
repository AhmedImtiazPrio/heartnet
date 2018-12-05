cd '/media/taufiq/Data1/heart_sound/heartnet/featGen'

wav_directory = 'wav';
file_list = struct2table(dir(wav_directory));
rootpath = char(file_list.folder(1));
file_list = file_list.name(file_list.isdir == 0);

data=[];
labels = importfile('wav.tsv');

for i=1:1
    [data,fs] = audioread([rootpath '/' char(file_list(i))]);
    length(data)
    
end


