function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

for i = 1:m
  % Compute train/cross validation errors using training examples 
  % X(1:i, :) and y(1:i), storing the result in 
  % error_train(i) and error_val(i)
  X_ = X(1:i, :);
  y_ = y(1:i);
  [theta] = trainLinearReg(X_, y_, lambda);

  [J, grad] = linearRegCostFunction(X_, y_, theta, 0);
  error_train(i, 1) = J;

  [J, grad] = linearRegCostFunction(Xval, yval, theta, 0);
  error_val(i, 1) = J;
end

end
