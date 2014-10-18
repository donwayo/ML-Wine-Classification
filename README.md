Description


Type classification:

We use naive bayesian binary classification. As for the quality classification, we split the dataset into training set (70%) and validation set (30%). We calculate the maximum likelihood estimator for the prior probability, the mean and the standard deviation of both red and white samples, and use them to calculate the discriminant function. The discriminant funciton gives us the probability of a sample being red or white wine. We classify the validation set according to which discriminant is bigger and calculate the error by estimating the mean square error of our prediction with the true values from the validation set. The only drawback for this algorithm is that it takes one feature into consideration to classify the type.


Quality classification

Trained a Gaussian MLE for each level, and each parameter. Thus resulting in generating 7x11x2 distributions for the classification. Each distribution was tested against the training set, and the ones that yielded the lowest error were selected for each quality level. This was then tested against the validation set so as to perform cross validation. 
