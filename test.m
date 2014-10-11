W = readtable('train_data.csv');
wine_type = zeros(size(W));
type_data = W{1:4000, 13};
strcmp(type_data, 'White')
