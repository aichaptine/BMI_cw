clc; clear all; close all;
load('monkeydata_training.mat')
[X_move, y_move, id_move] = preprocess_movement_data(trial, 3, 20);
[X_plan, y_plan, id_plan] = preprocess_planning_data(trial);

[X_move, y_move, id_move] = shuffle_data(X_move, y_move, id_move);
[X_plan, y_plan, id_plan] = shuffle_data(X_plan, y_plan, id_plan);

%%

L = 750;
acc = 0;
N=3;
for i = 1:N
    [X_move, y_move, id_move] = shuffle_data(X_move, y_move, id_move);
    [X_plan, y_plan, id_plan] = shuffle_data(X_plan, y_plan, id_plan);
    tree = fitctree(X_plan(:, 1:L)', y_plan(:, 1:L)');
    [Ynew, score, ~, ~] = predict(tree, X_plan(:, L+1:end)');

    acc = acc + sum(y_plan(L+1:end)'==Ynew) / length(Ynew);
end
acc = acc/N;

%need to split data into train, crossval, test.
%then do normalisation on training data; use mean/std from training to also
%normalise crossval and test.

%Lots of firing rates over 20ms intervals are 0 as no spikes
%occur (see single columnn of X). Is this a problem
