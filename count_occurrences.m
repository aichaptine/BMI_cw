function predicted_angles = count_occurrences(out)

%The input of this function is a matrix of the predicted angles of the test data having gone through for each model. 
%At each trial, the most occurrent angle is the predicted angle.
%The function outputs a vector of predicted angles for each trial.

    y = zeros(1,8);
    predicted_angles = zeros(1,size(out,2));
    
    for i=1:size(out,2)
        v = out(:,i);
        for j=1:length(y)
            y(j) = sum(v==j);
        end
        [argvalue, argmax] = max(y);
        predicted_angles(i) = argmax;
    end
end

