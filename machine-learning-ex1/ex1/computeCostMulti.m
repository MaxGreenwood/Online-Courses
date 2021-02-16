function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
theta = zeros(size(X, 2), 1);

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.



% define hypothesis function (m-dimensional)
h = X * theta;	
				
% find residual vector (m-dimensional)
residual = h - y; 				

% square each element
residualsquared = residual.^2;		

% sum the elements for J
J = (1/(2*m))*sum(residualsquared);


% =========================================================================

end
