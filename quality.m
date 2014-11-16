wine_data = readtable('train_data.csv');

% Select training data.
X = wine_data{1:4000, 1:11};
X_val = wine_data{4001:end, 1:11};

% Select labels
r = wine_data{1:4000, 12};
r_val = wine_data{1:4000, 12};

% Select features