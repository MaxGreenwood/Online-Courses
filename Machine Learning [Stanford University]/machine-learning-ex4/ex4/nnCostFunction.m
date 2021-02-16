function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



% Part 1

%%%% Unroll parameters 
%%%% nn_params = [Theta1(:) ; Theta2(:)];

% Add column of ones to data matrix
X1 = [ones(m, 1) X];

% Compute activation values for the hidden layer (25 x 5000)
a2 = sigmoid(Theta1 * X1');

% Add column of ones to hidden layer matrix
u = size(a2', 1);
a2_temp = [ones(u, 1) a2'];

% Compute output values (10 x 5000)
a3 = sigmoid(Theta2 * a2_temp');

% Translate observed y values into classif'n matrix (10 x 5000)
yClass = zeros(m,num_labels);
for i=1:m
 yClass(i,y(i)) = 1;
end 
yClass = yClass';

% Compute cost function (excl. regularisation)
residual = (-yClass.*log(a3))-((1-yClass).*log(1-a3));
J = sum((1/m)*sum(residual));


% Part 3

% Code regularisation term
regTerm = (lambda/(2*m))*(sum(sum(Theta1.^2)) + sum(sum(Theta2.^2)));

% Account for bias
biasTerm = (lambda/(2*m))*(sum(Theta1(:,1).^2) + sum(Theta2(:,1).^2));

% Add regularisation to cost function
J = J + regTerm - biasTerm;


% Part 2

for t = 1:m,

% Step 1: run feedforward pass for training examples t

% Select example, transpose to column vector and add bias unit
a1 = X(t,:)';
a1 = [1; a1];

% Compute z2 (25 x 1)
z2 = Theta1 * a1;

% Compute a2 and add bias unit (26 x 1)
a2 = sigmoid(z2);
a2 = [1 ; a2];

% Compute z3 (10 x 1)
z3 = Theta2 * a2;

% Compute a3 (10 x 1)
a3 = sigmoid(z3);


% Step 2: obtain cost error in output, delta3

% Take observed output from hypothesis output for the example
delta3 = a3 - yClass(:,t);


% Step 3: obtain cost error in hidden layer, delta2

% Compute delta for mapping from hidden to output (26 x 1)
delta2 = Theta2' * delta3 .* sigmoidGradient([1; z2]);


% Step 4: accumulate the gradient

% Remove delta2 zero (25 x 1)
 delta2 = delta2(2:end);

% Setup for accumulation of deltas
dt2 = delta3 * a2';
dt1 = delta2 * a1';

% Compute gradients
Theta2_grad = Theta2_grad + dt2;
Theta1_grad = Theta1_grad + dt1;

end


% Step 5: obtain the unregularised gradient for the cost function

Theta2_grad = (1/m) * Theta2_grad;
Theta1_grad = (1/m) * Theta1_grad;

% Step 6: account for regularisation

% Add the regulraisation terms, excluding j=0 index
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda/m) * Theta2(:,2:end));
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda/m) * Theta1(:,2:end));


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
