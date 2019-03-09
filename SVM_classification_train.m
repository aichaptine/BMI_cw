function [model_1_2, model_1_3, model_1_4, model1_5] = SVM_multiclass_Train(data)

%loop to go through all angles
for i=1:7
  for j= i+1:8
    
   model_i_j= svmTrain(data_for_angle_i_and_j, y, C, @linearKernel)
 



[X,y,id] = preprocess_planning_data(data);

xi = X(:,find(y==1)); %This is done within the loop for the choice of i and j
xj = X(:,find(y==2));

model = SVM(xi,xj);
