function predicted_angles = count_occurrences(out)
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

