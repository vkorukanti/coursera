function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

m = size(X,1);

% You need to return the following variables correctly.
idx = zeros(m, 1);

for x=1:m,
  diff = X(x, :) - centroids(1, :);
  minDist = diff * diff';
  idx(x, 1) = 1;
  for k=2:K,
    diff = X(x, :) - centroids(k, :);
    diff = diff * diff';
    if diff < minDist
      minDist = diff;
      idx(x, 1) = k;
    end;
  end;
end;

end

