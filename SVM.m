function model = SVM(xi, xj, sigma, C)

X = [xi xj];
y = [ones(1,length(xi)) zeros(1,length(xj))];

model = svmTrain(X', y',C, @(x1, x2) gaussianKernel(x1, x2, sigma));

end
