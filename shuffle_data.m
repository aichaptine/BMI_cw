function [X_out, y_out, id_out] = shuffle_data(X, y, id)
%Takes input training data (X - matrix of firing rates, y - corresponding
%labels, id - corresponding trial ids) and shuffles all in same way

    idx = randperm(size(y,2)); %generate shuffled indexes
    X_out = X(:, idx); %shuffle columns (each column is a training example)
    y_out = y(:, idx);
    id_out = id(:, idx);
end
