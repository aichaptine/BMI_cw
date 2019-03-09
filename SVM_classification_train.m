
[X,y,id] = preprocess_planning_data(data);

xi = X(:,find(y==1)); %This is done within the loop for the choice of i and j
xj = X(:,find(y==2));

model = SVM(xi,xj);
