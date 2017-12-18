function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

if (1 == 0) % remove if to find the C, sigma

vals = [0.01 0.03 0.1 0.3 1, 3, 10, 30];
error = zeros(8, 8);
for C_=1:8,
  for sigma_=1:8,
    C = vals(1, C_);
    sigma = vals(1, sigma_);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    error(C_, sigma_) = mean(double(predictions ~= yval));
  end;
end;

low_C=0;
low_sigma=0;
lowest_error = 297349023742037492;
for C_=1:8,
  for sigma_=1:8,
    if (lowest_error > error(C_, sigma_))
      low_C = C_;
      low_sigma = sigma_;
      lowest_error = error(C_, sigma_);
    end;
  end;
end;

C = vals(1, low_C);
sigma = vals(1, low_sigma);
else
C = 1;
sigma = 0.1;
endif

end
