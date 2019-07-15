function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top K eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% Z = zeros(size(X, 1), K);

% U has dimensions n x n

Ured = U(:,1:K);        % This has dimensions n x k

Z = X * Ured;
% Matrix dimensions:
% (m x k) = (m x n) * (n x k)


end
