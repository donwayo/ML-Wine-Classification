W = readtable('train_data.csv');
T = readtable('test_dataset.csv');

% [pred_naive_type, FScore_naive_type] = naive_bayes(W, 7, T)

% [pred_naive_qlt, FScore_naive_qlt] = naive_bayes_quality(W, T)

% [pred_grad, FScore_grad] = gradient_descent

f = generalized_linear_model(W, T);
% [pred_glm_type, FScore_glm_type, pred_log_qlt, FScore_log_qlt] = generalized_linear_model(W, T);

% [pred_svm,  confus,numcorrect,precision,recall,FScore_svm] = support_vector_machine(W,T);

% [pred_knn, FScore_knn] = k_nearest_neighbor(W,T);