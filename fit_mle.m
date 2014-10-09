function d = fit_mle(data,r,p)
    idx_p = find(r == p);
    
    p = length(idx_p) / length(data);
    m = mean(data(idx_p));
    s = std(data(idx_p));
    
    d = @(x) -0.5*log(2*pi) - log(s) - ((x - m).^2)/(2*s^2)+log(p);
end
