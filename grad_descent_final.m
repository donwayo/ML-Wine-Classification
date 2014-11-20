clear all 
close all

W = readtable('test_dataset.csv');
[N, d] = size(W);

wine = W(randperm(height(W)),:);
x_data = wine{:,1:d-2};
label_qlt =  wine{:,d-1};       % column d=12 for the quality
label_type = wine{:,d};         % column d=13 for the type
actual = strcmp(label_type, 'White');  % if wine is white, give value 1

mdl_type = fitglm(x_data, actual, 'Distribution', 'Binomial');
[pred_type, score] = predict(mdl_type, x_data);

mdl_qlt =  fitglm(x_data, label_qlt, 'Distribution', 'Poisson');
[pred_qlt, score2] = predict(mdl_qlt, x_data);

predicted = (pred_type > 0.5);
[confus,numcorrect,precision,recall,FScore]= getcm(actual, predicted, [0, 1])
[confus2,numcorrect2,precision2,recall2,FScore2]= getcm(label_qlt, round(pred_qlt), [1, 2, 3, 4, 5, 6, 7])
cross_val_mdl_type = crossval(mdl_type, 'KFold', 40)
kloss = kfoldLoss(cross_val_mdl_type)