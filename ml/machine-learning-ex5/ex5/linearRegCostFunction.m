function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

hx = X*theta;
err = hx - y;
err_sq = err .* err;
J = (1/(2*m))*sum(err_sq(:));

%create a clone of theta vector where theta(1,1) is 0
theta_clone = theta;
theta_clone(1,1) = 0;

% add regularization factor
reg = (lambda/(2*m))*sum((theta_clone .* theta_clone)(:));
J += reg;

grad = (1/m) * (X'*(hx - y)) + (lambda/m)*theta_clone;

end
