function model = SVM_train(xi, xj, sigma, C)

% binary svm classification

X = [xi xj]; %keeps only the training data relevant to this angle pair
y = [ones(1,size(xi,2)) zeros(1,size(xj,2))];  %processes the train data's labels: transforming to 1s for angle i and 0s for angle j

model = svmTrain(X', y',C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 

end
