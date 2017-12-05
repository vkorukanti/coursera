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

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');

h = h2;

pos_h = log(h);
neg_h = log(ones(size(h)) - h);

y_roll = zeros(m, num_labels);
for i=1:m,
  j = y(i,1);
  y_roll(i, j) = 1;
end;

pos_y = y_roll;
neg_y = ones(size(y_roll)) - y_roll;

J = (-1/m)*(sum((pos_y .* pos_h)(:)) + sum((neg_y .* neg_h)(:)));

% add regularization

% copy theta vectors and zero out bias values
Theta1_reg = Theta1;
Theta2_reg = Theta2;
Theta1_reg(:,1) = 0;
Theta2_reg(:,1) = 0;

reg = (lambda/(2*m))*( sum((Theta1_reg .* Theta1_reg)(:)) + sum((Theta2_reg .* Theta2_reg)(:)));

J = J + reg;


% gradients
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for t = 1:m,
 a1 = [1 X(t, :)]';
 z2 = Theta1 * a1;
 a2 = [1; sigmoid(z2)];
 z3 = Theta2 * a2;
 a3 = sigmoid(z3);

 delta_3 = a3 - y_roll(t, :)';
 delta_2 = Theta2' * delta_3 .* sigmoidGradient([1; z2]);
 
 delta_2 = delta_2(2:end); %skil delta_3_0 (bias)

 Theta2_grad = Theta2_grad + delta_3 * a2';
 Theta1_grad = Theta1_grad + delta_2 * a1';
end


Theta1_grad = Theta1_grad * (1/m);
Theta2_grad = Theta2_grad * (1/m);

Theta1_grad = Theta1_grad + (lambda/m)*Theta1_reg;
Theta2_grad = Theta2_grad + (lambda/m)*Theta2_reg;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
