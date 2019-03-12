function [X_train_out, y_train_out, X_test, y_test] = kfoldsplit(data, k, mode)
%Data is 100x8 struct
%Function performs data splitting for regression and classification

    %For classification:
    %1. Take 20 trials per reaching angle (RA) and place in test set; remainder in test set
    %2. Process the train and test data
    %3. Shuffle data
    %4. Normalise the train and test data
    %5. Split train set into k cross-validation sets
    %Output is of form:
    %X_train_out size (98 x N x k), where each face (98 x N) is a cross-val training set, and there are k of these
    %Same format for y_train_out, size (N x k), where each (1 x N) is a cross-val set, and there are k of these
    
    if (strcmp(mode,'classification')) %if splitting for classification
        %---------------------- 1&2. SPLIT & PROCESS TRAIN/TEST -------------------
        X_train = [];
        y_train = [];
        X_test = [];
        y_test = [];
        
        for angle = 1:8 %iterate through each angle
            test_idx = randsample(100,20); %randomly choose 20 trials without replacement
            train_idx = setdiff([1:1:100], test_idx); %train idx is numbers not choses
           
            [X_temp, y_temp, ~] = preprocess_planning_data(data([test_idx], angle)); %process this sub-selection
            X_test = [X_test, X_temp]; %append prev result to training data
            y_test = [y_test, y_temp];
           
            [X_temp, y_temp, ~] = preprocess_planning_data(data([train_idx], angle)); %process this sub-selection
            X_train = [X_train, X_temp];
            y_train = [y_train, y_temp];
        end
           
        %--------------------------- 3. SHUFFLE ------------------------
        [X_train, y_train, ~] = shuffle_data(X_train, y_train, zeros(size(y_train)));
        [X_test, y_test, ~] = shuffle_data(X_test, y_test, zeros(size(y_test)));

        %--------------------------- 4. NORMALISE ------------------------
        mx = mean(X_train, 2);
        sx = std(X_train, 0, 2);
        my = mean(y_train, 2);

        X_train = (X_train-mx)./sx;
        y_train = y_train - my;
        X_train((sx==0), :) = []; %remove rows for which standard deviation is 0 (since dont contribute to prediction)
        
        %--------------------------- 5. TRAIN/CV SPLIT ------------------------
        %CV data is placed in a 3d array (see top)
        cv_size = floor(length(y_train)/k); %divide training set into k equally sized blocks
        
        X_train_out = zeros( size(X_train,1), cv_size, k );
        y_train_out = zeros( cv_size, k );
        for i = 1:k
            X_train_out(:,:,i) = X_train(:, 1+(i-1)*cv_size : i*cv_size);
            y_train_out(:,i) = y_train(:, 1+(i-1)*cv_size : i*cv_size);
        end
        return
        
    elseif (strcmp(mode,'regression'))
        
        return
        
    else
        error('Incorrect "mode" argument to function - enter either "regression" or "classification"')
    end
end

