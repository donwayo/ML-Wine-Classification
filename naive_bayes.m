function [pred_ch, FScore] = naive_bayes(train_set, feat, test_set)

[N,M] = size(train_set); 

% a = 1;
% K = 2;  % number of classes

label = strcmp(train_set.type, 'White');	% assign 0/1 for red/white
whites = train_set(strcmp(train_set.type, 'White'),:);      % takes all the information for the white wines
x_white = whites{:,1:M-2} ;
reds = train_set(strcmp(train_set.type, 'Red'),:);  % takes all the information for the red wines
x_red = reds{:,1:M-2} ;

mle_p_white = numel(find(label == 1)) / N;
mle_m_white = sum( x_white(:,feat) )/ (N*mle_p_white);
mle_s_white = sqrt( mean( (x_white(:,feat)-mle_m_white).^2 ) );

mle_p_red = numel(find(label == 0)) / N;
mle_m_red = sum( x_red(:,feat) )/ (N*mle_p_red);
mle_s_red = sqrt( mean( (x_red(:,feat)-mle_m_red).^2 ) );

[num, d] = size(test_set);
discr_red = - log(2*pi)/2 - log(mle_s_red) - ((test_set{:, feat}-mle_m_red).^2)/(2*(mle_s_red^2)) + log(mle_p_red);
discr_white = - log(2*pi)/2 - log(mle_s_white) - ((test_set{:, feat}-mle_m_white).^2)/(2*(mle_s_white^2)) + log(mle_p_white);
pred_ch = discr_red < discr_white;

actual = strcmp(test_set{:,d}, 'White');
[confus,numcorrect,precision,recall,FScore]= getcm(actual, pred_ch, [0, 1]);
