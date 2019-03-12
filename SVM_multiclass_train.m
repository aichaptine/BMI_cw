
function [models] = SVM_multiclass_train(X_train,y_train, sigma, c)

% we aim to determine the reaching angle using the first 300ms. 
% the input is the training data and output is 28 models. e.g. model for angle 1 vs angle 2
% We use a one vs one multiclass SVM classification

%Pairwise method
%loop to go through all angles
m = 0;

  for i=1:7
    for j= i+1:8
    m = m+1;
    xi = X_train(:,find(y_train==i));
    xj = X_train(:,find(y_train==j));
    models(m) = SVM_train(xi, xj,sigma,c);
    end
  end
