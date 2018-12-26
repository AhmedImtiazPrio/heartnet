% clear; 
file_list = struct2table(dir('mfcc'));
rootpath = char(file_list.folder(1));
file_list = file_list.name(file_list.isdir == 0);
data = table();
names = table();
for i=1:length(file_list)
    loadpath = [rootpath '/' char(file_list(i))];
    try
        htkdata = array2table(htkread(loadpath));
        data = [data;htkdata];
        
        filename = char(file_list(i));
        filename = strsplit(filename(end-30:end),'.');
        filename = string(char(filename(end-2)));
        names = [names; table(repmat(filename,size(htkdata,1),1))];
    catch
        disp(i)
    end

end
names.filenames = names.Var1;
names.Var1 = [];
lldDataset = [names data];
% writetable(lldDataset,'mfcc.csv','Delimiter',';')
