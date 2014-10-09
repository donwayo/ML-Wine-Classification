W = readtable('train_data.csv');

[N,M] = size(W); 
n_train = round(N*0.7);
a = 1;
K = 2;  % number of classes

wine = W(randperm(height(W)),:);
train_set = wine(1:n_train,:);
valid_set = wine(n_train+1:end, :);

r = W(1:n_train,13);
x = wine{1:n_train, 1:11};

reds = train_set(strcmp(train_set.type, 'Red'),:);
x_red = reds{:,1:11} ;
% r_red = reds{:,feat};
r_red = zeros(1,height(reds));     % if wine is red, give value 0
whites = train_set(strcmp(train_set.type, 'White'),:);
x_white = whites{:,1:11} ;
% r_white = whites{:,feat};
r_white = ones(1,height(whites));  % if wine is white, give value 1
r_bl = [r_red r_white];

mle_pr = sum(r_bl) / n_train
% mle_p_red = (a + sum(r_red*x_red)) / (K*a + sum(r_red))
feat = 2;
mle_p_white = sum(r_white*x_white(:,feat))/ sum(r_white);
for i=1:11
    mle_p_red(i) = (a + sum(r_red*x_red(:,i))) / (K*a + sum(r_red));
    mle_p_white(i) = sum(r_white*x_white(:,i))/ sum(r_white);
end
mle_p_red, mle_p_white
% mle_p_red = sum( r_red*x_red ) / sum(r_red)
% mle_p_white = (a + sum(r_white*x_white))/ (K*a + sum(r_white))
% mle_p_white = sum( r_white*x_white ) / sum(r_white)
% mle_p_red = height(reds)/n_train;
% mle_p_white = height(whites)/n_train;

% w_0 = sum( log(1-mle_p_0j) ) + log(1-mle_pr) - sum( log(1-mle_p_1j) ) - log(mle_pr) 
%     % w_0 =   -2.4689 - 3.1416i
% w = sum( log( mle_p_0j*(1-mle_p1j) ) ) - sum( log( mle_p_1j*(1-mle_p0j) ) )
%     % w =    1.0447e+02 + 1.1938e+02i
