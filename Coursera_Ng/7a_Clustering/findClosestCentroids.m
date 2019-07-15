function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Initialize K
K = size(centroids, 1);

idx = zeros(size(X,1), 1);

dist = zeros(K,1);      % Preallocate memory for each example's distance from centroids

for i=1:size(X,1)
   for k=1:K
      dist(k) = sum((X(i,:) - centroids(k,:)).^2); 
   end
   [~ , min_ind] = min(dist);
   idx(i) = min_ind;
end


end

