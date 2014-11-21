% KNN Quality
clear all 
close all

W = readtable('train_data.csv');
[N, d] = size(W);

wine = W(randperm(height(W)),:);
ox_data = wine{:,1:d-2};

[coeff, score, latent, tsquared, explained, mu] = pca(ox_data);

pca_cols = 10;

%x_data = ox_data;
x_data = ox_data * coeff(:,1:pca_cols);

label_qlt =  wine{:,d-1};       % column d=12 for the quality
label_type = wine{:,d};         % column d=13 for the type

mdl_quality = fitcknn(x_data, label_qlt, 'NumNeighbors', 5, ...
    'NSMethod','exhaustive','Distance','mahalanobis','BreakTies', 'smallest');

rloss = resubLoss(mdl_quality);
kloss_best = 1;
neigh_best = 0;



for i = 1:15
    mdl_quality.NumNeighbors = i;
    cross_val_mdl_quality = crossval(mdl_quality, 'KFold', 40);
    kloss = kfoldLoss(cross_val_mdl_quality);
    %if(kloss < kloss_best)
    if(kloss - kloss_best < -0.005)
        neigh_best = i;
        kloss_best = kloss;
    end
    fprintf('Neighbors: %g, %g, %g\n', i, kloss, resubLoss(mdl_quality));
end

mdl_quality.NumNeighbors = neigh_best

% compare predicted labels with the actual labels
% and find the number of true elements

% ---------------------------------------------------
%           *** DELETE BEFORE SUBMISSION ***
% ---------------------------------------------------
% Try to classify with challenge data. 
CW = readtable('test_dataset.csv');

ox2_data = CW{:,1:11};
label_qlt = CW{:,12};

%x2_data = ox2_data;
x2_data = ox2_data * coeff(:,1:pca_cols);

p_labels = predict(mdl_quality,x2_data);

[a,b,c,d,F] = getcm(label_qlt, p_labels, [1,2,3,4,5,6,7]);
F
%[coeff, score, latent, tsquared, explained, mu] = pca(x_data);
