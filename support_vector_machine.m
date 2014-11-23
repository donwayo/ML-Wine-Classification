function [predicted, FScore] = support_vector_machine(TRAIN, TEST)

[N, d] = size(TRAIN);

x_train = TRAIN{:,1:d-2};
% train_qlt =  wine{:,d-1};       % column d=12 for the quality
train_type = TRAIN{:,d};         % column d=13 for the type

x_test = TEST{:,1:d-2};

mdl_type = fitcsvm(x_train, train_type, 'KernelFunction', 'linear', ...
    'Standardize', true, 'ClassNames', {'Red', 'White'})
[pred_type, score] = predict(mdl_type, x_test);

cross_val_mdl_type = crossval(mdl_type, 'KFold', 40);
kloss = kfoldLoss(cross_val_mdl_type);

actual = strcmp(TEST{:,d},'White');
predicted = strcmp(pred_type,'White');
[confus,numcorrect,precision,recall,FScore]= getcm(actual, predicted, [0, 1]);

figure;
h(1:2) = gscatter(x_test(:,1),x_test(:,2),TEST{:,d}, 'gr');
title('Scatterplot of Wine Type Classification using Support Vector Machine')