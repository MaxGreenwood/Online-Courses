function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


% Starting with an data set of m*n, X, append a column of ones to the left to make the data matrix X 
%%%%%%%%%   X = [ones(m, 1) X];

% Initialise theta (n+1-dimensional vector)
%%%%%%%%%   theta = zeros(size(X)(1,2),1);

% Define hypothesis function (m-dimensional vector)
h = sigmoid(X * theta);

% Define residuals (m-dimensional vector)
residual = (-y.*log(h))-((ones(size(y))-y).*log(1-h));

% Define J
J = (1/m)*sum(residual);

% Initialise grad
grad = zeros(size(theta));

% Define gradient of J for each feature
grad = (1/m)*(X'*(h-y));




% =============================================================

end
