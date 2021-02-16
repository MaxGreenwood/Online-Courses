function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

for j = 1:size(X, 2)

% calculate mean of j features and put them in row vector
mu(1, j) = mean(X(:,j));

% calculate std of j features and put them in row vector
sigma(1, j) = std(X(:,j));

% enlarge these vectors to matrix of size X (copy rows i times)
for i = 1:size(X, 1)
	MU(i,:) = mu;
	SIGMA(i,:) = sigma;
end;

% obtain the normalised data matrix
X_norm = (X - MU) ./ SIGMA;

% ============================================================

end
