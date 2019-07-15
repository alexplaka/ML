function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

C_array = [0.01 0.03 0.1 0.3 0.65 1 2 3];
sigma_array = [0.01 0.03 0.65 0.1 0.15 0.3 0.65 1];

Error = zeros(size(C_array,2) , size(sigma_array,2));

for i = 1:size(C_array,2)
    for j = 1:size(sigma_array,2)
        model = svmTrain(X,y,C_array(i),@(x1, x2) gaussianKernel(x1, x2, sigma_array(j)));
        predictions = svmPredict(model, Xval);
        Error(i,j) = mean(double(predictions ~= yval));
    end
end

[MinError , ind] = min(Error(:));

[C_ind , sigma_ind] = ind2sub(size(Error), ind);

C = C_array(C_ind);
sigma = sigma_array(sigma_ind);


end
