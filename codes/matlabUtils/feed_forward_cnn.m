function res = feed_forward_cnn(s,parms)
    nfreqbands = size(s,2);
    flat_out = [];
    for fb=1:nfreqbands
        prms.H1 = squeeze(parms.H1(fb,:,:));
        prms.H2 = squeeze(parms.H2(fb,:,:,:));
        prms.b1 = parms.b1(fb,:);
        prms.b2 = parms.b2(fb,:);
        prms.maxpooling = parms.maxpooling;
        out = feed_forward_filters(s(:,fb),prms);
        flat_out = [flat_out; flatten(out)];
    end
    res = flat_out.'*parms.W1 + parms.bias1;
    res(res<=0) = 0; %relu
    res = res*parms.W2 + parms.bias2;
    res = sigmoid(res);
end

function y = sigmoid(x)
    y = 1./(1+exp(-x));
end

function res = flatten(x)
    res = nan(numel(x),1);
    [~,j] = size(x);
    res(1:j:end) = x(:,1);
    res(2:j:end) = x(:,2);
    res(3:j:end) = x(:,3);
    res(4:j:end) = x(:,4);
end
function out2 = feed_forward_filters(s,parms)
    [nfilters, filter_sz] = size(parms.H1);
    out1 = nan(max(length(s)-max(0,filter_sz-1),0)/2,nfilters);
    for nf=1:nfilters
        tmp = conv(s,parms.H1(nf,:),'valid') + parms.b1(nf);
        tmp(tmp<=0) = 0; %relu
        tmp = reshape(tmp,parms.maxpooling,length(tmp)/parms.maxpooling);
        tmp = max(tmp)';
        out1(:,nf) = tmp;
    end

    [nfilters1,nfilters2,filter_sz] = size(parms.H2);
    L = max(size(out1,1)-max(0,filter_sz-1),0);
    out2 = nan(L/2,nfilters1);
    for k=1:nfilters1
        h = squeeze(parms.H2(k,:,:));
        tmp = 0;
        for nf2=1:nfilters2
            tmp = conv(out1(:,nf2),h(nf2,:),'valid') + tmp;
        end
        tmp = tmp + parms.b2(k); % add bias
        tmp(tmp<=0) = 0; %ReLu
        tmp = reshape(tmp,parms.maxpooling,length(tmp)/parms.maxpooling);
        tmp = max(tmp)';
        out2(:,k) = tmp;
    end


end