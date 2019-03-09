function model = SVM_train(xi, xj, sigma, C)

X = [xi xj];
y = [ones(1,size(xi,2)) zeros(1,size(xj,2))];

model = svmTrain(X', y',C, @(x1, x2) gaussianKernel(x1, x2, sigma));

end
