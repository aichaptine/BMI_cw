function [models] = SVM_multiclass_train(data)

sigma=0.1;
C=1.0;
models = zeros(1,28);

[X,y,id] = preprocess_planning_data(data);

%Pairwise method
%loop to go through all angles
for v=1:length(models)
  for i=1:7
    for j= i+1:8
    xi = X(:,find(y==i));
    xj = X(:,find(y==j));
    models(v) = SVM_train(xi, xj,sigma,c);
    end
  end
end
  
