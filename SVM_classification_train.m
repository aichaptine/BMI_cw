function [model_1_2, model_1_3, model_1_4, model1_5] = SVM_multiclass_Train(data)

sigma=0.1;
C=1.0;

[X,y,id] = preprocess_planning_data(data);

%Pairwise method
%loop to go through all angles
for i=1:7
  for j= i+1:8
  xi = X(:,find(y==i));
  xj = X(:,find(y==j));
  model = SVM_train(xi, xj,sigma,c);
  end
end
  
