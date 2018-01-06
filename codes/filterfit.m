%%filterfit
clear
load('/media/taufiq/Data/heart_sound/feature/LTSA.mat')
close all
% for n=20:60
n=37;
b=cell(1,5);
for i=1:6
    if i==2    % continue if littmann
        continue;
    end
    b{i}=firls(n,F(1:end-1)./500,normal(2,1:end-1)./normal(i,1:end-1));
    fvtool(b{i},1)
    title(['filter' num2str(i)])
    set(gca,'xscale','log')
end
%% plot results
close all
clc
for i=1:6
    if i==2
        continue;
    end
    figure;
    h = freqz(b{i},1,F,1000);
    plot(F,abs(h))
    hold on
    plot(F,normal(i,:))
    plot(F,normal(i,:).*abs(h))
    plot(F,normal(2,:))
%     plot(F,normal(2,:)./normal(i,:))
    set(gca,'xscale','log')
    set(gca,'yscale','log')
    xlim([F(1) F(end)])
    ylim([10^-4 10])
    grid on
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
    title(['Freq Characteristics after applying gain with Filter ' num2str(i) ' with order ' num2str(n)]);
    legend('Filter Gain', ['a'+i-1 ' without gain'], ['a'+i-1 ' with gain'], 'Littmann')
    hold off
    %% MSE
    MSE(i)=sum((normal(i,:).*abs(h)-normal(2,:)).^2)/length(normal(i,:));
    disp(['For filter ' num2str(i)])
    disp(MSE(i))
end
error(n)=sum(MSE);
% end