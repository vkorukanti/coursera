function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

centroids = zeros(K, n);
cluster_sizes = zeros(K, 1);

for x=1:m,
  c = idx(x, 1);
  centroids(c, :) = centroids(c, :) + X(x, :);
  cluster_sizes(c, 1) = cluster_sizes(c, 1) + 1; 
end;

for k=1:K,
  centroids(k, :) = centroids(k, :) / cluster_sizes(k, :);
end;

end

