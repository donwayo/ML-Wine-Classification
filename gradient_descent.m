function [y_ch, FScore] = gradient_descent

[NUM,TEXT,RAW]  = xlsread('train_data.csv');

FixedAcidity = NUM(:,1);
VolatileAcidity = NUM(:,2);
CitricAcid = NUM(:,3);
ResidualSugar = NUM(:,4);
Chlorides = NUM(:,5);
FreeSulfurDioxide = NUM(:,6);
TotalSulfurDioxide = NUM(:,7);
Density = NUM(:,8);
Ph = NUM(:,9);
Sulphates = NUM(:,10);
Alcohol = NUM(:,11);
Quality = NUM(:,12);
Type = TEXT([2:length(TEXT)],13);


[m,n] = size(NUM);
%Converting matrix Type from string to number 0/1 whether the type is White or
%Red respectively
y = strcmp(Type,'White');
%c=2;

red = find(y==0);
white = find(y==1);

figure (1)

plot(FixedAcidity(red), VolatileAcidity(red), '.r');
hold on
plot(FixedAcidity(white), VolatileAcidity(white), '.', 'MarkerEdgeColor', [1, 0.84, 0]);
title('Scatterplot of Wine Type Classification using Gradient Descent')



% Gradient descent with maximum likelihood estimation
Nit = 1000;
theta(1,:) = zeros(1,n);
alfa = 0.01;

% FEATURE SCALING
% Given a new x value we must first normalize x using the mean and standard deviation
% that we had previously computed from the training set.


x = NUM(:,1:n-1);

eta_value = mean(x);
sigma_value = std(x);

eta = repmat(mean(x),m,1);

sigma = repmat(std(x),m,1);

x = (x - eta)./sigma;

x = [ones(m,1), x];

for i=2:Nit+1

% h_theta = sigmoid(x*theta(i-1,:)');
% J_der = 1/m*(h_theta - y)'*x;
% theta(i,:) = theta(i-1,:) - alfa*J_der;
% J(i-1) = computeCost(y,h_theta);

[J(i-1),J_der] = costFunction(theta(i-1,:),x,y);
theta(i,:) = theta(i-1,:) - alfa*J_der;

end

theta_opt = theta(Nit+1,:);

figure (2)
plot(J);
title('Cost function in Gradient descent')

figure (3)
plot(theta(:,1));

figure (4) 
plot(theta(:,2));

[NUM_CH,TEXT_CH,RAW_CH]  = xlsread('test_dataset.csv');

x_ch = NUM_CH(:,1:n-1);

[m_ch,n_ch] = size(x_ch);

eta_ch = repmat(mean(x_ch),m_ch,1);

sigma_ch = repmat(std(x_ch),m_ch,1);

x_ch = (x_ch - eta_ch)./sigma_ch;

x_ch = [ones(m_ch,1) x_ch];

h_theta_ch = sigmoid(x_ch*theta_opt');

y_ch = round(h_theta_ch);

actual = strcmp(TEXT_CH([2:length(TEXT_CH)],13), 'White');

[confus,numcorrect,precision,recall,FScore]= getcm(actual, y_ch, [0, 1])