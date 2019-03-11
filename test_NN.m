clc; clear all; close all;
load('monkeydata_training.mat')

%Take only one reaching angle and train model
%N.B. only took first 80 trials for training; remainder are for testing
[X_move, y_move, id_move] = preprocess_movement_data(trial(1:80,1), 3, 20);

%[X_move, y_move, id_move] = preprocess_movement_data(trial, 3, 20);
[X_plan, y_plan, id_plan] = preprocess_planning_data(trial);

[X_move, y_move, id_move] = shuffle_data(X_move, y_move, id_move);
[X_plan, y_plan, id_plan] = shuffle_data(X_plan, y_plan, id_plan);

%% PREDICT REACHING ANGLE FROM 300ms INITIAL DATA (will be used to select regression model)
% L = 750;
% acc = 0;
% N=3;
% for i = 1:N
%     [X_move, y_move, id_move] = shuffle_data(X_move, y_move, id_move);
%     [X_plan, y_plan, id_plan] = shuffle_data(X_plan, y_plan, id_plan);
%     tree = fitctree(X_plan(:, 1:L)', y_plan(:, 1:L)');
%     [Ynew, score, ~, ~] = predict(tree, X_plan(:, L+1:end)');
% 
%     acc = acc + sum(y_plan(L+1:end)'==Ynew) / length(Ynew);
% end
% acc = acc/N;

%% NORMALISE DATA
X_train = X_move;
y_train = y_move;

mx = mean(X_train, 2);
sx = std(X_train, 0, 2);
my = mean(y_train, 2);

X_train = (X_train-mx)./sx;
y_train = y_train - my;
X_train((sx==0), :) = [];


%% TRAIN NN, 100 HIDDEN UNITS
net = feedforwardnet(100);
[net, tr] = trainscg(net, X_train, y_train)

%% TEST ON SPECIFIC TRIAL (I.E. NOT IN TRAINING SET)
%test on trial (M,1)
X_t = [];
y_t = [];
L=20;
B=3;
M=88;
movement_spikes = trial(M,1).spikes(:,300:end); %300 or 301?
movement_handPos = trial(M,1).handPos(:,300:end);
T = size(movement_spikes, 2); %gives length of cut sample

for k = B*L:L:T
    spike_sample = movement_spikes(:, k+1-B*L : k);

    %calculate features (firing rates) for all 98 neurons for each bin
    feature_sample = splice_data(spike_sample, B);

    %calculate change in x,y position over 20ms (L) interval
    delta_handPos_sample = movement_handPos(1:2, k) - movement_handPos(1:2, k+1-L); 

    %append above to the output matrices
    feature_sample = (feature_sample - mx)./sx;
    X_t = [X_t, feature_sample];
    y_t = [y_t, delta_handPos_sample];
end 

%NEED TO MAKE SURE WE REMOVE ROWS OF TEST DATA THAT WE REMOVED FOR TESTING
X_t((sx==0),:) = [];

y_pred = net(X_t);
y_pred = y_pred + my; %undo zero-centre output

figure()
hold on
plot(y_pred(1,:))
plot(y_t(1,:))
hold off

pos_pred = cumsum(y_pred,2);
pos_true = cumsum(y_t,2);

figure()
[theta,rho] = cart2pol(pos_pred(1,:), pos_pred(2,:));
polarplot(theta,rho)
hold on
[theta,rho] = cart2pol(pos_true(1,:), pos_true(2,:));
polarplot(theta,rho)
legend('Predicted', 'True')
hold off
