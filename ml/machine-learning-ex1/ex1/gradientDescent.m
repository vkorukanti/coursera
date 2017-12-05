function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    theta_correction = zeros(2, 1);
    for i=1:m,
        hi = theta'*X(i,:)' - y(i,1);
        theta_correction(1,1) = theta_correction(1,1) + hi;
        theta_correction(2,1) = theta_correction(2,1) + hi * X(i,2);
    end;
    theta_correction = theta_correction*alpha/m;
    theta = theta - theta_correction;

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

end
