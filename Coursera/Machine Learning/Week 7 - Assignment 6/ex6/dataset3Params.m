function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
CtoTry = [.01, .03, .1, .3, 1, 3, 10, 30];
%CtoTry = [.25 .5 1 5 10];
sigmatoTry = CtoTry;
%CtoTry = 1;
%sigmatoTry = .3;
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
errorTable = zeros(numel(CtoTry),numel(sigmatoTry)); %(i,j)->C[i], sigmatoTry[j]
for k = 1:numel(CtoTry)
   for l = 1:numel(sigmatoTry)
      model = svmTrain(X, y, CtoTry(k), ...
      @(x1,x2) gaussianKernel(x1,x2, sigmatoTry(l)), 1e-3,20);
      predictions = svmPredict(model, Xval);
      errorTable(k,l) = mean(double(predictions ~= yval));
   end
end
[val, idx] = min(errorTable(:));
[minCIdx, minsigmaIdx] = ind2sub(size(errorTable),idx);
C = CtoTry(minCIdx);
sigma = sigmatoTry(minsigmaIdx);
% =========================================================================

end
