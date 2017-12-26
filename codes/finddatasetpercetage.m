clear
fold=3;
a=0;
b=0;
c=0;
d=0;
e=0;
f=0;
train=importlist(['train' num2str(fold) '.txt']);
val= importlist(['validation' num2str(fold) '.txt']);
train=sort(train);
val=sort(val);
%%
for i=1:length(train)
    temp = char(train(i));
    switch temp(1)
        case 'a'
            a=a+1;
        case 'b'
            b=b+1;
        case 'c'
            c=c+1;
        case 'd'
            d=d+1;
        case 'e'
            e=e+1;
        case 'f'
            f=f+1;
    end
end
disp("Percentages")
a/length(train)*100
b/length(train)*100
c/length(train)*100
d/length(train)*100
e/length(train)*100
f/length(train)*100
%%
a=0;
b=0;
c=0;
d=0;
e=0;
f=0;

for i=1:length(val)
    temp = char(val(i));
    switch temp(1)
        case 'a'
            a=a+1;
        case 'b'
            b=b+1;
        case 'c'
            c=c+1;
        case 'd'
            d=d+1;
        case 'e'
            e=e+1;
        case 'f'
            f=f+1;
    end
end
disp("Percentages")
a/length(val)*100
b/length(val)*100
c/length(val)*100
d/length(val)*100
e/length(val)*100
f/length(val)*100

