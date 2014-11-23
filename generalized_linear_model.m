function [predicted_type, FScore, predicted_qlt, confus2,numcorrect2,precision2,recall2, FScore2] = generalized_linear_model(TRAIN, TEST)
[N, d] = size(TRAIN);

x_train = TRAIN{:,1:d-2};
qlt_label =  TRAIN{:,d-1};       % column d=12 for the quality
type_label = TRAIN{:,d};         % column d=13 for the type
type_train = strcmp(type_label, 'White');  % if wine is white, give value 1

x_test = TEST{:,1:d-2};
qlt_test = TEST{:,d-1};

mdl_type = fitglm(x_train, type_train, 'Distribution', 'Binomial', 'Link', 'logit');
[pred_type, score] = predict(mdl_type, x_test);

% figure(1);
% w = [1 0.85 0];
feat1 = 7; feat2 = 11;       % select features to plot with

% h(1:2) = gscatter(x_test(:,feat1),x_test(:,feat2),TEST{:,d}, 'cr');
% title('Scatterplot of Wine Type Classification using Generalized Logistic Regression')

mdl_qlt =  fitglm(x_train, qlt_label, 'Distribution', 'Poisson', 'Link', 'log');
[pred_qlt, score2] = predict(mdl_qlt, x_test);

actual = strcmp(TEST{:,d}, 'White');
predicted_type = (pred_type > 0.5);
predicted_qlt = round(pred_qlt);

[confus,numcorrect,precision,recall,FScore]= getcm(actual, predicted_type, [0, 1])
[confus2,numcorrect2,precision2,recall2,FScore2]= getcm(qlt_test, predicted_qlt, [1, 2, 3, 4, 5, 6, 7]);

% cross_val_mdl_type = crossval('mcr', x_test, actual, 'Predfun', predicted_type)
% kloss = kfoldLoss(cross_val_mdl_type)