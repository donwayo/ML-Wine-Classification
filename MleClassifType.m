W = readtable('train_data.csv');

[N,M] = size(W); 
n_train = round(N*0.7);
a = 1;
K = 2;  % number of classes

wine = W(randperm(height(W)),:);
train_set = wine(1:n_train,:);
valid_set = wine(n_train+1:end, :);

feat = 2;
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
mle_p_red = sum(x_red(:,feat)) / length(r_red)
mle_p_white = sum(x_white(:,feat)) / length(r_white)