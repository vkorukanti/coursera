function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

hx = sigmoid(X*theta);
pos = log(hx);
neg = log(ones(size(hx)) - hx);
J = (-1/m)*(sum((y' * pos)(:)) + sum(((1-y)'*neg)(:)));

%create a clone of theta vector where theta(1,1) is 0
theta_clone = theta;
theta_clone(1,1) = 0;

% add regularization factor
reg = (lambda/(2*m))*(sum((theta_clone .* theta_clone)(:)));
J += reg;

grad = (1/m) * (X'*(hx - y)) + (lambda/m)*theta_clone;

end
