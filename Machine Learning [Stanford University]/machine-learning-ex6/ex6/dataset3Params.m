function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Define values
values = [0.01 0.03 0.1 0.3 1 3 10 30];

% Initialise error to large value
err = 1000;

% Write for loop to cycle through values for C and sigma

for i=1:length(values)
  
  for j=1:length(values)
    
    C1 = values(i);
    sigma1 = values(j);
    
    % Set a model in the SVM training function with values C1 and sigma1
    model = svmTrain(X, y, C1, @(x1, x2) gaussianKernel(x1, x2, sigma1));
    predictions = svmPredict(model, Xval);
    
    err1 = mean(double(predictions ~= yval));
    % Switch in values of C and sigma if lower error is produced
    if err1 < err
      C = C1;
      sigma = sigma1;
      err = err1;
      
      % Tell me what these good values are
      fprintf('new min found C, sigma = %f, %f with error = %f', C1, sigma1, err1)
      
    endif
    
  endfor
  
endfor

C1 = C;
sigma1 = sigma;

% =========================================================================

end
