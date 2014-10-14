function [p, m, s, p2, m2, s2] = fit_mle(data,r,pos)
    idx_p = find(r == pos);
    idx_n = find(not(r == pos)); 
    
    p = length(idx_p) / length(data);
    m = mean(data(idx_p));
    s = std(data(idx_p));
    
    p2 = length(idx_n) / length(data);
    m2 = mean(data(idx_n));
    s2 = std(data(idx_n));
    
    
    %d = @(x) -0.5*log(2*pi) - log(s) - ((x - m).^2)/(2*s^2)+log(p);
end
