function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%








% Starting with an data set of m*n, X, append a column of ones to the left to make the data matrix X 
%%%%%%%%%   X = [ones(m, 1) X];

% Initialise theta (n+1-dimensional vector)
%%%%%%%%%   theta = zeros(size(X)(1,2),1);

% Define hypothesis function (m-dimensional vector)
h = sigmoid(X * theta);

% Define residuals (m-dimensional vector)
residual = (-y.*log(h))-((ones(size(y))-y).*log(1-h));

% Define unregularised J
unregJ = (1/m)*sum(residual);

% Initialise grad
grad = zeros(size(theta));

% Define gradient of unregJ for each feature
grad = (1/m)*(X'*(h-y));

% Define regularisation term
regTerm = (sum(lambda*(theta.^2))/(2*m));

% Account for the fact that the first indexed parameter is not regularised
regTerm1 = (lambda*(theta(1)^2))/(2*m);

% Define J
J = unregJ + regTerm - regTerm1;

% Define length of theta to help treat theta(1) differently
k = length(theta);

% Define gradient of J for each feature (n+1-dimensional vector)
grad(1,1) = (X(:,1))'*(h-y)./m;
grad(2:k,1) = ((X(:,2:k))'*(h-y)./m) +(lambda*theta(2:k,1))./m;





% =============================================================

grad = grad(:);

end
