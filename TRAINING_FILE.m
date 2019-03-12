clear all;
load('monkeydata_training.mat');

%% Classifying different reaching angles using the 300ms before movement.

sigma=500;
c=1.0; %Cross-validation to find best values for these may be needed

[X,y,id] = preprocess_planning_data(trial);  %preprocess the data, returns: firing rate for first 300ms in 'X', 
                                            %Takes reaching angle for trial in 'y'  and Takes trial id in 'id'
%[X,y,id] = shuffle_data(X, y, id);
X_train = X(:, 1:640); 
y_train = y(:, 1:640);
X_test =  X(:, 641:end);
y_test =  y(:, 641:end);
out = zeros(28,size(X_test, 2));
models = SVM_multiclass_train(X_train,y_train,sigma,c);

for i=1:length(models)
    out(i,:) = svmPredict(models(i), X_test');
end

predictions = convert_to_RA(out);
predicted_angle = count_occurrences(predictions); %Counts the number of occurrence of each 
dist = predicted_angle-y_test;
error = 1-(sum(dist==0)/length(dist));
accuracy = 1-error;
