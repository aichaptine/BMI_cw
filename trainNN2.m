clc; clear all; close all;
load('monkeydata_training.mat')

%Take only one reaching angle and train model
%N.B. only took first 80 trials for training; remainder are for testing
[X_train, y_train, id_move] = preprocess_movement_data(trial(1:80,7), 3, 20);
[X_train, y_train, ~] = shuffle_data(X_train, y_train, id_move);

mx = mean(X_train, 2);
sx = std(X_train, 0, 2);
my = mean(y_train, 2);

X_train = (X_train-mx)./sx;
y_train = y_train - my;
X_train((sx==0), :) = [];


%%
input_layer_size = size(X_train,1);
net1 = constructNet(0.1,1,0,input_layer_size,100,2)

%%
net1 = trainNet(X_train, y_train, net1, 50, 32);

%%
y_pred = predictNet(X_train(:,1:50), net1)
%%
function net = constructNet(lr, b, lambda, I, H, O);

    e = 0.1;
    theta1 = rand(H, 1 + I) * 2 * e - e;
    theta2 = rand(O, 1 + H) * 2 * e - e;
    
    net = struct('lr', lr, 'beta', b, 'lambda', lambda, 'input_size', I, ...
        'hidden_size', H, 'output_size', O, 'theta1', theta1, 'theta2', theta2, 'performance', 0);
    
end

function net_out = trainNet(X, y, net_in, E, B)
    
    for epoch = 1:E
        %need to shuffle?
        for sample = 1:B:length(y)-B
            X_batch = X(:, sample:sample+B-1);
            y_batch = y(:, sample:sample+B-1);
            
            [dw1, dw2] = backwardPass(X_batch, y_batch, net_in, B);
            
            net_in.theta1 = net_in.theta1 + dw1; %update weights of net
            net_in.theta2 = net_in.theta2 + dw2;
        end
        fprintf('Epoch = %d', epoch)
        perf = estimatePerformance(X, y, net_in)
    end
    net_out = net_in;
    
    function MSE = estimatePerformance(X_test, y_test, net) 
        [~,~,y_pred] = forwardPass(X_test, net);
        MSE = 0;
        for i = 1:size(y_test, 2)
            MSE = MSE + norm(y_test(:,i) - y_pred(:,i))^2;
        end
    end
    function [w1, w2] = backwardPass(X, y, net, B)
        %get weights and fwd propagate
        lr = net.lr;
        beta = net.beta;
        lambda = net.lambda;
        theta1 = net.theta1;
        theta2 = net.theta2;
        
        [V0, V1, V2] = forwardPass(X, net);

        %back prop - CHECK
        delta2 = (y - V2);
        z2 = theta2'*delta2;
        z2 = z2(2:end,:);
        u1 = theta1*V0;
        delta1 = sigmoidGrad(u1, beta) .* z2;

        %delta1 = (1/(2*B)) * sum(delta1,2);
        %delta2 = (1/(2*B)) * sum(delta2,2);
        w2 = (1/(2*B))*(lr * delta2 * V1') - (2 * lr * lambda * theta2);
        w1 = (1/(2*B))*(lr * delta1 * V0') - (2 * lr * lambda * theta1);
    end
    function [V0, V1, V2] = forwardPass(X, net)
        beta = net.beta;
        theta1 = net.theta1;
        theta2 = net.theta2;

        m = size(X,2); %get number of samples 
        V0 = [ones(1,m); X]; %add bias term

        %Hidden layer = sigmoid units
        u1 = theta1*V0;
        V1 = sigmoid(u1, beta);

        %Output layer = linear units
        V1 = [ones(1,m); V1]; %add bias term
        u2 = theta2*V1;
        V2 = u2;
    end
    function f = sigmoid(u, b)
        %performs sigmoid function of u with parameter b(eta)
        %u may be any dimension
        f = zeros(size(u));
        f = 1.0 ./ (1.0 + exp(-b*u));
    end
    function g = sigmoidGrad(u, b)
        %calculates gradient of sigmoid at u with parameter b(eta)
        g = zeros(size(u));
        for i = 1:size(u,1)
            for j = 1:size(u,2)
                f = sigmoid(u(i,j), b);
                g(i,j) = b * f * (1-f);
            end
        end
    end
end

function y_pred = predictNet(X, net_in)
    [~,~,y_pred] = forwardPass(X, net_in)
    function [V0, V1, V2] = forwardPass(X, net)
        beta = net.beta;
        theta1 = net.theta1;
        theta2 = net.theta2;

        m = size(X,2); %get number of samples 
        V0 = [ones(1,m); X]; %add bias term

        %Hidden layer = sigmoid units
        u1 = theta1*V0;
        V1 = sigmoid(u1, beta);

        %Output layer = linear units
        V1 = [ones(1,m); V1]; %add bias term
        u2 = theta2*V1;
        V2 = u2;
    end
    function f = sigmoid(u, b)
        %performs sigmoid function of u with parameter b(eta)
        %u may be any dimension
        f = zeros(size(u));
        f = 1.0 ./ (1.0 + exp(-b*u));
    end
    function g = sigmoidGrad(u, b)
        %calculates gradient of sigmoid at u with parameter b(eta)
        g = zeros(size(u));
        for i = 1:size(u,1)
            for j = 1:size(u,2)
                f = sigmoid(u(i,j), b);
                g(i,j) = b * f * (1-f);
            end
        end
    end
end

function 