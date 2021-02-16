function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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


% Add more polynomials by employing the mapFeature function
% Note that mapFeature also adds a column of ones for us, so the intercept term is handled
%%%%% X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
%%%%%%initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
%%%%% lambda = 1;

% Define hypothesis function (m-dimensional vector)
h = sigmoid(X * theta);

% Define residual sum term
residualTerm = (-((y'*log(h))+((1-y)'*log(1-h)))./m);

% Define regularisation term
regTerm = (sum(lambda*(theta.^2))/(2*m));

% Account for the fact that the first indexed parameter is not regularised
regTerm1 = (lambda*(theta(1)^2))/(2*m);

% Define J
J = residualTerm + regTerm - regTerm1;

% Define length of theta to help treat theta(1) differently
k = length(theta);

% Define gradient of J for each feature (n+1-dimensional vector)
grad(1,1) = (X(:,1))'*(h-y)./m;
grad(2:k,1) = ((X(:,2:k))'*(h-y)./m) +(lambda*theta(2:k,1))./m;



% =============================================================

end
