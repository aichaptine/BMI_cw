clc; clear all; close all;
load('monkeydata_training.mat')

%Take only one reaching angle and train model
%N.B. only took first 80 trials for training; remainder are for testing
[X_train, y_train, id_move1] = preprocess_movement_data(trial(1:80,7), 3, 20);
[X_cv, y_cv, id_move2] = preprocess_movement_data(trial(81:end,7), 3, 20);
[X_train, y_train] = shuffleData(X_train, y_train);
[X_cv, y_cv] = shuffleData(X_cv, y_cv);

mx = mean(X_train, 2);
sx = std(X_train, 0, 2);
my = mean(y_train, 2);

X_train = (X_train-mx)./sx;
X_cv = (X_cv-mx)./sx;
y_train = y_train - my;
y_cv = y_cv - my;
X_train((sx==0), :) = [];
X_cv((sx==0), :) = [];

%%
function modelParameters = positionEstimatorTraining(training_data)
