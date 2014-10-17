clear all 
close all

W = readtable('train_data.csv');
[N, d] = size(W);

wine = W(randperm(height(W)),:);
n_train = round(N*0.7);
train_set = wine(1:n_train,:);
valid_set = wine(n_train+1:end, :);

r_train = wine(1:n_train,d);       % column d=13 for the type
x_train = wine{1:n_train,1:d-2};
r_type = strcmp(train_set.type, 'White');  % if wine is white, give value 1

x_valid = W{n_train+1:end,1:d-2};
r_valid = strcmp(valid_set.type, 'White');

niter = 1000;       % number of iterations
theta = zeros(1, (d-2));
alpha = 0.001;        % step

eta = repmat(mean(x_train), n_train, 1);    % feature scaling
sigma = repmat(std(x_train), n_train, 1);
x_train = (x_train - eta)./sigma;

for i = 2:niter+1

    h_theta = 1./(1+exp(-x_train*theta'));
    J_der = -( h_theta - r_type )' * x_train/n_train;
    theta = theta - alpha*J_der;
    J(i-1) = (r_type' * log(h_theta) + (1-r_type') * log(1-h_theta))/n_train;

end

theta_opt = theta ;

h_theta_val = 1./( 1+exp(-x_valid*theta_opt') );
 
pred = (h_theta_val >= 0.5);

err = -(r_valid'*log(h_theta_val) + (1-r_valid')*log(1 - h_theta_val));

CH = readtable('challenge_data.csv');
[N_ch, d_ch] = size(CH);
x_ch = CH{:,2:d_ch-2};

eta_ch = repmat(mean(x_ch), N_ch, 1);    % feature scaling
sigma_ch = repmat(std(x_ch), N_ch, 1);
x_ch = (x_ch - eta_ch)./sigma_ch;

h_theta_ch = 1./( 1+exp(-x_ch*theta_opt') );
y_ch = (h_theta_ch >= 0.5);
white = 100*sum((y_ch == 1))/N_ch
red = 100*sum((y_ch == 0))/N_ch

CH_new = [CH array2table(y_ch)];