% ----------------------------------------------------------
%    @TODO: we need to specify 'polynomialOrder' kernel
%       if we want to use SVM for the quality classification
% ----------------------------------------------------------

clear all 
close all

W = readtable('test_dataset.csv');
[N, d] = size(W);

wine = W(randperm(height(W)),:);
x_data = wine{:,1:d-2};
label_qlt =  wine{:,d-1};       % column d=12 for the quality
label_type = wine{:,d};         % column d=13 for the type

mdl_type = fitcsvm(x_data, label_type, 'KernelFunction', 'linear', ...
    'Standardize', true, 'ClassNames', {'White', 'Red'})
[pred_type, score] = predict(mdl_type, x_data);

% % see @TODO
% mdl_qlt = fitcsvm(x_data, label_qlt, 'KernelFunction', 'polynomialOrder', ...
%     'Standardize', true)

cross_val_mdl_type = crossval(mdl_type, 'KFold', 40)
kloss = kfoldloss(cross_val_mdl_type)
% compare predicted labels with the actual labels
% and find the number of true elements
accur = numel(find(strcmp(pred_type, label_type) == 1))/length(label_type)*100

% ---------------------------------------------------
%           *** DELETE BEFORE SUBMISSION ***
% ---------------------------------------------------
% The code below is to split dataset in training and validation
% since we're using cross-validation, we don't split the dataset
% 
% n_train = round(N*0.7);
% 
% x_train = wine{1:n_train,1:d-2};
% train_type = table2cell(wine(1:n_train,d));     
% train_qlt = wine{1:n_train,d-1};                
% 
% x_valid = wine{n_train+1:end,1:d-2};
% valid_type = wine{n_train+1:end,d};
% valid_qlt= wine{n_train+1:end,d-1};
% 
% mdl_type = fitcsvm(x_train, train_type, 'KernelFunction', 'linear', ...
%     'Standardize', true, 'ClassNames', {'White', 'Red'})
% [pred_type, score] = predict(mdl_type, x_valid);
% 
% cross_val_mdl_type = crossval(mdl_type, 'KFold', 40)

% accur = numel(find(strcmp(pred_type, valid_type) == 1))/length(valid_type)*100