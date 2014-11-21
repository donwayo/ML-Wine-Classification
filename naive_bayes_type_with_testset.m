W = readtable('train_data.csv');

[N,M] = size(W); 
n_train = round(N*0.7);
a = 1;
K = 2;  % number of classes

wine = W(randperm(height(W)),:);
train_set = wine(1:n_train,:);
valid_set = wine(n_train+1:end, :);
feat = 5   % based on which feature we want to make the prediction

reds = train_set(strcmp(train_set.type, 'Red'),:);  % takes all the information for the red wines
x_red = reds{:,1:M-2} ;
r_red = zeros(1,height(reds));     % if wine is red, give value 0
whites = train_set(strcmp(train_set.type, 'White'),:);      % takes all the information for the white wines
x_white = whites{:,1:M-2} ;
r_white = ones(1,height(whites));  % if wine is white, give value 1

mle_p_red = sum(x_red(:,feat)) / length(r_red);
mle_m_red = sum( x_red(:,feat) )/length(r_red);
mle_s_red = sqrt( mean( (x_red(:,feat)-mle_m_red).^2 ) );
mle_p_white = sum(x_white(:,feat)) / length(r_white);
mle_m_white = sum( x_white(:,feat) )/length(r_white);
mle_s_white = sqrt( mean( (x_white(:,feat)-mle_m_white).^2 ) );

% === Take the validation set ===

discr_red = - log(2*pi)/2 - log(mle_s_red) - ((wine{n_train+1:end, feat}-mle_m_red).^2)/(2*(mle_s_red^2)) + log(mle_p_red);
discr_white = - log(2*pi)/2 - log(mle_s_white) - ((wine{n_train+1:end, feat}-mle_m_white).^2)/(2*(mle_s_white^2)) + log(mle_p_white);
pred = discr_red < discr_white;
wine_type = strcmp(valid_set.type, 'White');    % assign 0/1 for red/white
err = mean((wine_type-pred).^2)

% === Take the validation set ===

CH = readtable('test_dataset.csv');
[num, d] = size(CH);
discr_red = - log(2*pi)/2 - log(mle_s_red) - ((CH{:, feat}-mle_m_red).^2)/(2*(mle_s_red^2)) + log(mle_p_red);
discr_white = - log(2*pi)/2 - log(mle_s_white) - ((CH{:, feat}-mle_m_white).^2)/(2*(mle_s_white^2)) + log(mle_p_white);
pred_ch = discr_red < discr_white;
white = 100*sum(pred_ch == 1)/num
red = 100*sum(pred_ch == 0)/num
wine_type2 = strcmp(CH.type, 'White');    % assign 0/1 for red/white

[confus,numcorrect,precision,recall,FScore]= getcm(wine_type2, pred_ch, [0, 1])

% CH_new = [CH array2table(pred_ch)];
% csvwrite('challenge_classified.csv', CH_new)


