function g = classify_mle(data, ds)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

logp = zeros(length(data),length(ds));

for d = 1:length(ds)
    
    mle = ds{d};
    
    p = mle(:,1);
    m = mle(:,2);
    s = mle(:,3);
    p2 = mle(:,4);
    m2 = mle(:,5);
    s2 = mle(:,6);
    
    g_p = -0.5*log(2*pi) - log(s) - ((data - m).^2)/(2*s^2)+log(p);
    g_n = -0.5*log(2*pi) - log(s2) - ((data - m2).^2)/(2*s2^2)+log(p2);
    
    logp(:,d) = g_p > g_n;
end

%[o, g] = max(logp,[],2);

g = logp;

end

