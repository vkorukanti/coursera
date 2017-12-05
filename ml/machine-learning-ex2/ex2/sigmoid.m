function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

g = ones(size(z)) ./ (ones(size(z)) + exp(-z));

end
