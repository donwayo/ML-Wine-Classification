wine_data = readtable('train_data.csv');
chall_data = readtable('challenge_data.csv');

wine_parameters_disc = cell(11,7);

wine_quality = [1,2,3,4,5,6,7];
wine_parameters = [1,2,3,4,5,6,7,8,9,10,11];

wine_quality_n = 7;
wine_parameters_n = 11;

% Select training data.
training_data = wine_data{1001:5000, 1:12};
validation_data = wine_data{1:1000, 1:12};

% Actual wine quality labels.
r = training_data(:,12);
r_val = validation_data(:,12);

r_q_idx = cell(wine_quality_n,1);



for i = wine_quality
    r_q_idx{i} = find(r == i);
end


% Classification results
g = cell(wine_parameters_n,1);
e = zeros(wine_parameters_n,1);
e_p = zeros(wine_parameters_n, wine_quality_n);

% Do for everything.
for j = wine_parameters
    % Select the data for one parameter.
    v_d = training_data(:,j);
    
    % Calculate MLE discriminators for the selected parameter.
    for i = wine_quality 
        [p, m, s, p2, m2, s2] = fit_mle(v_d, r, i);
        wine_parameters_disc{j,i} = [p,m,s,p2,m2,s2];
    end
    
    % Test the accuracy of each discriminator with the validation set.
    g{j} = classify_mle(v_d, wine_parameters_disc(j,:));
    
    
    for d = wine_quality
        rd = r == d;
        e_p(j,d) = mean( g{j}(:,d) == rd );
    end
end

fbest_param = zeros(11,2);
params = wine_parameters;


%% Selection
% Do this for each label of wine quality.

best_params_cells = cell(length(wine_quality),1);

for d = wine_quality
    
    % This is the expected r vector for each quality label.
    rd = r == d;
    
    % This is the accumulator for each quality label.
    accum = zeros(length(r),1);
    
    % Initialize variables.
    params = wine_parameters;
    b = 1;
    
    while not(isempty(params))
        % params is going to contain all the parameters to be tested
        % all of them are tested, and the best one is selected and 
        % removed from the 'params' vector.
        
        % Stores the best parameter, and the error associated.
        best_param = [0,0];
        
        % Take out each parameter after it is selected as the best for the
        % accumulated error.
        for p = 1:length(params)
            % Calculate the error
            e_acc = mean((accum | g{params(p)}(:,d)) == rd);
            
            % Check if it's the best yet.
            if e_acc >= best_param(2)
                best_param = [p, e_acc];
            end
        end
        % Save best parameter into the matrix
        fbest_param(b,:) = [params(best_param(1)),best_param(2)];
        b = b + 1;
        
        % Remove parameter from the vector.
        params(best_param(1)) = [];
        
        % Update the accumulator for quality label
        accum = accum | g{best_param(1)}(:,d);
    end
    
    % Save the best for this quality label.
    best_params_cells{d} = fbest_param;
end

%% Clasification 
%

vl_data = validation_data;
cl_data = chall_data{:, 1:12};


val_c = zeros(length(cl_data),7);
te_c = zeros(length(cl_data),7);

%best_params_cells{3}(1,:) = [4,0];

y = zeros(length(cl_data),1);

for i = 1:length(best_params_cells)
    best_param = best_params_cells{i}(1,1);
    mle_vars = wine_parameters_disc{best_param,i};
    val_c(:,i)= classify_mle(vl_data(:,j), {mle_vars}); 
    te_c(:,i) = classify_mle(cl_data(:,j), {mle_vars}); 
    
    mean(val_c(:,i) == (r_val == i))
    
end
