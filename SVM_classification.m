function [] = SVM_classification(data)

sigma=0.1;
C=1.0;

[X,y,id] = preprocess_planning_data(data);

xi = X(:,find(y==1));
xj = X(:,find(y==2));
X = [xi xj];
size(X)
y = [ones(1,length(xi)) zeros(1,length(xj))];
size(y)
model = svmTrain(X', y',C, @(x1, x2) gaussianKernel(x1, x2, sigma));



%visualizeBoundary(X, y, model);
end
