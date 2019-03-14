function [best_param_C,best_param_sigma]= opt_param(trial)
[X,y,id] = preprocess_planning_data(trial); 
X_train = X(:, 1:640); 
y_train = y(:, 1:640);
X_test =  X(:, 641:end);
y_test =  y(:, 641:end);
out = zeros(28,size(X_test, 2));
% k=20;
% mode='classification';
% [X_train, y_train, X_test, y_test] = kfoldsplit(trial, k, mode)

%
best_score=0;
best_param_C=0;
best_param_sigma=0;
y=logspace(-3,3,7);
%C=10
%y1=[C/3:C/3:C*3]

for l=1:length(y)
    for j=1:length(y)
        C_values=y(l);
        sigma_values=y(j);
        models = SVM_multiclass_train(X_train,y_train,sigma_values,C_values);
     for i=1:length(models)
            models = SVM_multiclass_train(X_train,y_train,sigma_values,C_values);
     end
            out(i,:) = svmPredict(models(i), X_test');
            predictions = convert_to_RA(out);
            predicted_angle = count_occurrences(predictions); %Counts the number of occurrence of each 
            dist = predicted_angle-y_test;
            error = 1-(sum(dist==0)/length(dist));
            accuracy = 1-error;
            score=accuracy; 
                if score>best_score
                   best_score = score;
                   best_param_C= C_values;
                   best_param_sigma=sigma_values;
 
                end
      score
      best_param_C
     C2=C_values
     S1=sigma_values
    end
end