load('monkeydata_training.mat');

%% Classifying different reaching angles using the 300ms before movement.

[X,y,id] = preprocess_planning_data(trial);
X_train = X(:, 1:640);
y_train = y(:, 1:640);
X_test =  X(:, 641:end);
y_test =  y(:, 641:end);
out = zeros(28,length(X_test));
models = SVM_multiclass_train(X_train,y_train);

for i=1:length(models)
    out(i,:) = svmPredict(models(i), X_test);
end
