function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

hx = sigmoid(X*theta);
pos = log(hx);
neg = log(ones(size(hx)) - hx);
J = (-1/m)*(sum((y' * pos)(:)) + sum(((1-y)'*neg)(:)));

grad = (1/m) * (X'*(hx - y));

end
