function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% Add ones to the X data matrix
a1 = [ones(m, 1) X];

% returns a matrix of size [size[a1, 1] size[Theta1, 1]] = m x 25  where 25 is
% number of units in second layer, m is number of training examples in X
z2 = a1 * Theta1';

a2 = sigmoid(z2);

% add bias variable
a2 = [ones(m, 1) a2];

% returns a matrix of [m size[Theta2, 1]]
z3 = a2 * Theta2';

a3 = sigmoid(z3);

[max max_indices] = max(a3, [], 2);
p = max_indices;

end
