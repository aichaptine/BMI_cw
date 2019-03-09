function [models] = SVM_multiclass_train(data)

% we aim to determine the reaching angle using the first 300ms. 
% the input is the training data and output is 28 models. e.g. model for angle 1 vs angle 2
% We use a one vs one multiclass SVM classification


sigma=0.1; C=1.0;  % variables for the svm classification
models = zeros(1,28); %initialising the output

[X,y,id] = preprocess_planning_data(data);  %preprocess the data, returns: firing rate for first 300ms in 'X', 
                                            %Takes reaching angle for trial in 'y'  and Takes trial id in 'id'


%Pairwise method
%loop to go through all angles

for v=1:length(models)
  for i=1:7
    for j= i+1:8
    xi = X(:,find(y==i));  %finds training data relevant to this angle pair
    xj = X(:,find(y==j));
    models(v) = SVM_train(xi, xj,sigma,c); %trains the data for this angle pair
    end
  end
end
  
