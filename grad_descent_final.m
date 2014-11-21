clear all 
close all

W = readtable('train_data.csv');
[N, d] = size(W);
TEST = readtable('test_dataset.csv');

wine = W(randperm(height(W)),:);
x_train = wine{:,1:d-2};
qlt_label =  wine{:,d-1};       % column d=12 for the quality
type_label = wine{:,d};         % column d=13 for the type
type_train = strcmp(type_label, 'White');  % if wine is white, give value 1

x_test = TEST{:,1:d-2};
qlt_test = TEST{:,d-1};

mdl_type = fitglm(x_train, type_train, 'Distribution', 'Binomial');
[pred_type, score] = predict(mdl_type, x_test);

figure;
h(1:2) = gscatter(x_test(:,1),x_test(:,2),TEST{:,d}, 'gr');

mdl_qlt =  fitglm(x_train, qlt_label, 'Distribution', 'Poisson');
[pred_qlt, score2] = predict(mdl_qlt, x_test);

actual = strcmp(TEST{:,d}, 'White');
predicted = (pred_type > 0.5);

[confus,numcorrect,precision,recall,FScore]= getcm(actual, predicted, [0, 1])
[confus2,numcorrect2,precision2,recall2,FScore2]= getcm(qlt_test, round(pred_qlt), [1, 2, 3, 4, 5, 6, 7])
% cross_val_mdl_type = crossval(mdl_type, 'KFold', 40)
% kloss = kfoldLoss(cross_val_mdl_type)