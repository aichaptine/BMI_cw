clc; clear all; close all;
load('monkeydata_training.mat')

%Take only one reaching angle and train model
%N.B. only took first 80 trials for training; remainder are for testing
[X_train, y_train, id_move1] = preprocess_movement_data(trial(1:80,7), 3, 20);
[X_cv, y_cv, id_move2] = preprocess_movement_data(trial(81:91,7), 3, 20);
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
input_layer_size = size(X_train, 1);
net1 = constructNet(0.1, 1, 1.2915, input_layer_size, 20, 2)

%%
[net1,p] = trainNet(X_train, y_train, X_cv, y_cv, net1, 10, 128);

%%
lr = logspace(-3,-1,5);
lambda = logspace(-3,1,10);
perf_cv = zeros(length(lr), length(lambda), 1, 341);
perf_tr = perf_cv;

for k = 20
    fprintf('\nK: %d, ', k)
    for i = 1:length(lr)
        fprintf('lr: %d, ', lr(i))
        for j = 1:length(lambda)
            fprintf('lambda: %d ', lambda(j))
            net1 = constructNet(lr(i),1,lambda(j),input_layer_size,k,2);
            [net1,p] = trainNet(X_train, y_train, X_cv, y_cv, net1, 20, 32);
            perf_cv(i,j,1,:) = p.cv;
            perf_tr(i,j,1,:) = p.train;
            clear net1
        end
    end
end
%%
y_pred = predictNet(X_train(:,1:50), net1)

%%
%%
L=20;
B=3;
for M = 60:100
    X_t = [];
    y_t = [];
    movement_spikes = trial(M,1).spikes(:,260:end); %300 or 301?
    movement_handPos = trial(M,1).handPos(:,260:end);
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

    y_pred = predictNet(X_t, net1);
    y_pred = y_pred + my; %undo zero-centre output

%     figure()
%     subplot(1,2,1)
%     hold on
%     plot(y_pred(1,:))
%     plot(y_t(1,:))
%     hold off
%     xlabel('Bin number')
%     ylabel('Predicted x movement')
%     subplot(1,2,2)
%     hold on
%     plot(y_pred(2,:))
%     plot(y_t(2,:))
%     hold off
%     legend('Predicted', 'True')
%     xlabel('Bin number')
%     ylabel('Predicted y movement')

    pos_pred = cumsum(y_pred,2);
    pos_true = cumsum(y_t,2);

    %figure()
    [theta,rho] = cart2pol(pos_pred(1,:), pos_pred(2,:));
    polarplot(theta,rho,'r')
    rlim([0 120])
    hold on
    polarplot(theta(end),rho(end),'*r')
    [theta,rho] = cart2pol(pos_true(1,:), pos_true(2,:));
    polarplot(theta,rho,'b')
    polarplot(theta(end),rho(end),'*b')
    legend('Predicted Trajectory', '', 'True Trajectory', '')
    hold off
    pause(1)
    %close all;
end
%%
function net = constructNet(lr, b, lambda, I, H, O)

    e = 0.05;
    theta1 = rand(H, 1 + I) * 2 * e - e;
    theta2 = rand(O, 1 + H) * 2 * e - e;
    
    net = struct('lr', lr, 'beta', b, 'lambda', lambda, 'input_size', I, ...
        'hidden_size', H, 'output_size', O, 'theta1', theta1, 'theta2', theta2, 'dropout1', [], 'dropout2', [], 'performance', 0);
    
end

function [net_out, perf_stats] = trainNet(X_train, y_train, X_cv, y_cv, net_in, E, B)
%     train_perf = zeros(1,E+1);
     test_perf = zeros(1, E+1);
%     cv_perf = zeros(1, E+1);
    
    train_perf = estimatePerformance(X_train, y_train, net_in);
    cv_perf = estimatePerformance(X_cv, y_cv, net_in);

    for epoch = 1:E
        %fprintf('\nEpoch = %d', epoch)
        fprintf('.')
        [X_train, y_train] = shuffleData(X_train,y_train);
        for sample = 1:B:length(y_train)-B
            X_batch = X_train(:, sample:sample+B-1);
            y_batch = y_train(:, sample:sample+B-1);
            
            %net_in = dropout(net_in,0.1,0.3);
            [dw1, dw2] = backwardPass(X_batch, y_batch, net_in, B);
            
            net_in.theta1 = net_in.theta1 + dw1; %update weights of net
            net_in.theta2 = net_in.theta2 + dw2;

            if rem(sample-1, 50*B) == 0
                train_perf = [train_perf estimatePerformance(X_train, y_train, net_in)];
                cv_perf = [cv_perf estimatePerformance(X_cv, y_cv, net_in)];
            end
        end
%         train_perf(epoch+1) = estimatePerformance(X_train, y_train, net_in);
%         cv_perf(epoch+1) = estimatePerformance(X_cv, y_cv, net_in);
    end
    
    figure();hold on; plot(train_perf); plot(cv_perf); hold off; legend('Train Loss', 'CV Loss')
    net_out = net_in;
    perf_stats.train = train_perf;
    perf_stats.cv = cv_perf;
    perf_stats.test = test_perf;
    
    function MSE = estimatePerformance(X_test, y_test, net) 
        [~,~,y_pred] = forwardPass(X_test, net);
        MSE = 0;
        m = size(y_test, 2);
        for i = 1:m
            MSE = MSE + norm(y_test(:,i) - y_pred(:,i))^2;
        end
        MSE = MSE/m;
    end
    function [w1, w2] = backwardPass(X, y, net, B)
        %get weights and fwd propagate
        lr = net.lr;
        beta = net.beta;
        lambda = net.lambda;
        
%         missing_inp_units = net.dropout1;
%         missing_hid_units = net.dropout2;
%         %dropout neurons
% 
%         net.theta1(:, missing_inp_units + 1) = 0;
%         net.theta1(missing_hid_units, :) = 0;
%         net.theta2(:, missing_hid_units + 1) = 0;

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
        theta1(:,1) = 0;
        theta2(:,1) = 0;
        w2 = (1/B)*(lr * delta2 * V1' - lr * lambda * theta2);
        w1 = (1/B)*(lr * delta1 * V0' - lr * lambda * theta1);
        
%         w1(:, missing_inp_units + 1) = 0;
%         w1(missing_hid_units, :) = 0;
%         w2(:, missing_hid_units + 1) = 0;
%         
%         w1 = w1*0.7;
%         w2 = w2*0.7;
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
    function net = dropout(net, p1, p2)
        I = net.input_size;
        H = net.hidden_size;
        
        r1 = rand(1,I);
        r2 = rand(1,H);
        
        idx1 = find(r1<p1);
        idx2 = find(r2<p2);
        
        net.dropout1 = idx1;
        net.dropout2 = idx2;
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
    [~,~,y_pred] = forwardPass(X, net_in);
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
        f = 1.0 ./ (1.0 + exp(-b*u));
    end
end

function [X_out, y_out] = shuffleData(X, y)
    idx = randperm(size(y,2)); %generate shuffled indexes
    X_out = X(:, idx); %shuffle columns (each column is a training example)
    y_out = y(:, idx);
end