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

C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
C_vec_len = size(C_vec, 1);
sigma_vec_len = size(sigma_vec, 1);
error_matrix = zeros(C_vec_len, sigma_vec_len);

for i = 1:C_vec_len
    for j = 1:sigma_vec_len
        C_curr = C_vec(i);
        sigma_curr = sigma_vec(j);
        model_curr = svmTrain(X, y, C_curr, @(x1, x2) gaussianKernel(x1, x2, sigma_curr));
        predictions_curr = svmPredict(model_curr, Xval);
        error_matrix(i,j) = mean(double(predictions_curr ~= yval));
    end
end

[~,I] = min(error_matrix(:));
[I_row, I_col] = ind2sub(size(error_matrix),I);
C = C_vec(I_row);
sigma = sigma_vec(I_col);

% =========================================================================

end
