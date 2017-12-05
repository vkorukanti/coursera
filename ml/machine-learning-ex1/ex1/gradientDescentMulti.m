function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta, 1);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    %theta_correction = zeros(size(theta));
    %for i=1:m,
    %    hi = theta'*X(i,:)' - y(i,1);
    %    for j=1:n,
    %        theta_correction(j, 1) = theta_correction(j,1) + hi * X(i,j);;
    %    end;
    %end;
    %theta_correction = theta_correction*alpha/m;
    %theta = theta - theta_correction;
    theta = theta - (alpha/m)*((X*theta - y)' * X)';

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
