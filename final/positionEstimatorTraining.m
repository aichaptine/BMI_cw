function modelParameters = positionEstimatorTraining(training_data)
%MODEL PARAMETERS STUCTURE:
%10x1 cell array
%[ {regressor_RA1}, {regressor_RA2}, ..., {regressor_RA8}, {SVM}, {reach_angle}]

%Regressor_RA:
%regressor_RAi is neural network model trained specifically for reaching
%angle i. It contains a struct model of a network with parameters:
%net = struct('lr', 'beta', 'lambda', 'input_size', 'hidden_size', 'output_size', 'theta1', 'theta2', 'performance', 'mx', 'sx', 'my');
%In training, theta1, theta2, mx, sx, my are calculated/learnt.
%The net struct is then stored in its entirety in the corresponding cell of modelParameters

%SVM:
%Please fill in

% %States:
% %States are of form:
% %states = struct('reach_angle', 0, 'current_pos', [0 0]);
% %These are passed as input to the testing algorithm to ensure that values are not repeatedly
% %re-calculated. The testing algorithm updates them at each call.
% %For example, the net outputs change in x,y for a given 60ms input. To
% %avoid calculating all previous changes in order to get current position,
% %this is instead passed in via the state cell.
% %Meanwhile, reaching angle only needs to be predicted on the first call of
% %positionEstimator.

num_angles = 8;
modelParameters = cell(num_angles+2,1);
modelParameters{end} = 0;

fprintf('Commencing training of models...\n')

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %       SVM CLASSIFIER TRAINING        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Training SVM.\n')
%[X_plan, y_plan] = preprocessPlanningData(training_data);
%[X_plan, y_plan] = shuffleData(X_plan, y_plan);

%Insert SVM training
%modelParameters{end-1} <- SVM parameters
fprintf('Finished Training SVM.\n\n')

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %        NN REGRESSOR TRAINING         %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%NEURAL NETWORK REGRESSOR OVERVIEW 
%Net is of form described above.

%Network input/output
%Net takes in feature vector calculated from previous 60ms of data (most 
%recent 20ms segment and previous 40ms) and outputs CHANGE in x,y positions
%over duration of most recent 20ms segment. Assumes that previous 40ms of
%spike data is relevant to current movement.

%Preprocessing of data [preprocessMovementData]:
%1. Take spike data from 261ms* to end-100ms of each trial, for a specific angle
%2. On this spike data, select a 20ms bin and take the 3 previous 20ms bins (inc. current),
%   and compute firing rates for each neuron in each bin. This will yield 3*98 firing
%   rates and will form the feature vector, X, for a single sample.
%3. X is then normalised by subtracting the mean and dividing by std.
%4. For the selected 20ms bin, compute change in positions to get y.
%5. y is then mean centred.
%6. A single iteration of the above forms a single sample, with length(X) = 3*98, and length(y) = 2
%7. Above is repeated by shifting current 20ms bin along by 1ms. This
%   yields approx 20K samples.
%*261 as assume movement starts in 301-320ms bin, and taking current &
%previous 2 bins (40ms overlal) of spike data as 'relevant' to movement

%Network Architecture:
%The network used is 3 layer (input, hidden, output).
%The hidden layer uses a sigmoidal activation function, while the output
%layer is linear.

%Network Training [trainNet]:
%Training is done using mini-batch (batch_size = 128) gradient descent with backpropagation
%The training set is looped over for a selected number of epochs.
%L2 regularization is used, with hyperparameter lambda.

        
num_angles = 8;
learn_rate = [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]; %need to optimise
reg_param = [3 3 3 3 3 3 3 3];

%FOR EACH ANGLE, BUILD A NN SEPARATE MODEL
for angle = 1:num_angles 
    fprintf('Training model #%d', angle)
    %Pre-process data (301ms onwards)      
    [X_move, y_move] = preprocessMovementData(training_data(:,angle), 3, 20);
    [X_move, y_move] = shuffleData(X_move, y_move); %shuffle samples

    %Normalise data
    mx = mean(X_move, 2);
    sx = std(X_move, 0, 2);
    my = mean(y_move, 2);
    X_move = (X_move-mx)./sx; %normalise x (feature - spike rates) data
    y_move = y_move - my; %mean centre y data (position changes per bin)
    X_move((sx==0), :) = []; %remove rows with 0 std as no variation so dont contribute

    %Construct Neural Network model
    input_layer_size = size(X_move, 1);
    %lr, beta, lambda, input layer size, hidden layer size, output layer
    %size
    net = constructNet(learn_rate(angle), 1, reg_param(angle), input_layer_size, 50, 2);
    net.mx = mx;
    net.sx = sx;
    net.my = my;
    
    %Train NN model
    %10 epochs, 100 batch size
    net = trainNet(X_move, y_move, net, 20, 128);
    modelParameters{angle} = net;
    clear net
    fprintf('\n')
end
fprintf('Finished training.\n')

function [X, y] = preprocessMovementData(data, B, L)
%Function takes in data structure (100x8 trials) and:
%1. Computes features -> X (where each column is a single training sample):
%example column of X: [----98 neuron FRs for bin 1----, ----98 neuron FRs for bin2----, ... ----98 neuron FRs for bin B----]
%i.e. column length is 98*B
%2. Compute target outputs -> y (change in hand pos over selected bin)
%3. Label each sample with trialId -> id (each column is trialID for
%corresponding sample)
%N.B data input is 100x8 array. Each element is a structure containing
%trialID, spikes (98xL), handPos (2xL)

    [N_trials, N_angles] = size(data);
    
    y = [];
    X = [];
    
    for i = 1:N_trials
        for j = 1:N_angles %access each cell
            
            %Take only data corresponding to movement
            movement_spikes = data(i,j).spikes(:, 301-B*L:end-100); %300 or 301?
            movement_handPos = data(i,j).handPos(:, 301-B*L:end-100);
            T = size(movement_spikes, 2); %gives length of cut sample
            
            %Pass along spike-trains, take multi-bin segments (length B*L). Then get
            %firing rates for each neuron in each bin (this will give 98*B firing rates i.e. a vector length 98*B. 
            %Then place these firing rates in X, change in handPos in y and trialId in id
            for k = B*L:T %need to start from B*L e.g. start at 60ms (if 3bins of size 20ms) since we need previous 60 values
                spike_sample = movement_spikes(:, k+1-B*L : k);
                
                %calculate features (firing rates) for all 98 neurons for each bin
                feature_sample = spliceData(spike_sample, B);
                
                %calculate change in x,y position over 20ms (L) interval
                delta_handPos_sample = movement_handPos(1:2, k) - movement_handPos(1:2, k+1-L); 
                
                %append above to the output matrices
                X = [X, feature_sample];
                y = [y, delta_handPos_sample];
            end
        end
    end
end
function feature_vector = spliceData(data, B)
%Takes in data segment length T ms (i.e. NxT) and
%1. Splits data into B segments (i.e segment size: N x T/B)
%2. Extracts firing rates from each segment
%Outputs 1-D vector of length (N*B x 1):
%[----fr_bin1----, ----fr_bin2----, ..., ----fr_binB----]
%e.g. fr_bin1 contains firing rates for neurons in 1st bin
    [N, T] = size(data);
    feature_vector = zeros(N*B, 1);
    L = T/B; %L = bin length
    
    for i = 1:B
        bin = data(:, 1+(i-1)*L : i*L); %take sub section of input data
        fr = compute_rate(bin); %fr will be size N
        feature_vector(1+(i-1)*N:i*N) = fr;
    end
    
    function firing_rate = compute_rate(spike_trains)
        % Takes in an NxT matrix containing spike-trains length 'sample_length' for 'num_trains'  different neural units
        % Outputs N-dim vector containing firing rate for each neural unit
        [num_trains, sample_length] = size(spike_trains);
        firing_rate = zeros(num_trains, 1);

        spike_count = sum(spike_trains, 2); %count number of spikes per spike-train
        firing_rate = spike_count*1000/sample_length; %divide by length of spike-train
    end
end
function [X, y] = preprocessPlanningData(data)
%Takes in 100x8 data structure, and for each trial
%1. Calc firing rates for first 300ms -> column of X
%2. Takes reaching angle for trial -> y
%3. Takes trial id -> id

    [N_trials, N_angles] = size(data);
    
    y = [];
    X = [];

    for i = 1:N_trials
            for j = 1:N_angles %access each trial

                %Take only data corresponding to planning (<300ms)
                planning_spikes = data(i,j).spikes(:,1:300); 
                T = 300; %gives length of cut sample
                
                %Calc firing rate for all neurons in 300ms interval
                spike_count = sum(planning_spikes, 2); %count number of spikes per spike-train
                firing_rates = spike_count*1000/T; %divide by length of spike-train

                X = [X, firing_rates]; %Column of firing rates for 98 neurons for single trial
                y = [y, j]; %Put corresponding target (reaching angle) in y
            end
        end
end
function net = constructNet(lr, b, lambda, I, H, O)
    e = 0.05;
    theta1 = rand(H, 1 + I) * 2 * e - e;
    theta2 = rand(O, 1 + H) * 2 * e - e;
    
    net = struct('lr', lr, 'beta', b, 'lambda', lambda, 'input_size', I, ...
        'hidden_size', H, 'output_size', O, 'theta1', theta1, 'theta2', theta2, 'performance', 0, 'mx', 0, 'sx', 0, 'my', 0);
    
end
function [net_out] = trainNet(X_train, y_train, net_in, E, B)

    for epoch = 1:E
        %fprintf('\nEpoch = %d', epoch)
        fprintf('.')
        [X_train, y_train] = shuffleData(X_train,y_train);
        for sample = 1:B:length(y_train)-B
            X_batch = X_train(:, sample:sample+B-1);
            y_batch = y_train(:, sample:sample+B-1);
            
            [dw1, dw2] = backwardPass(X_batch, y_batch, net_in, B);
            
            net_in.theta1 = net_in.theta1 + dw1; %update weights of net
            net_in.theta2 = net_in.theta2 + dw2;
        end
        estimatePerformance(X_train, y_train, net_in);
    end
    net_out = net_in;
    net_out.performance = estimatePerformance(X_train, y_train, net_out);
    
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

        theta1 = net.theta1;
        theta2 = net.theta2;
    
        [V0, V1, V2] = forwardPass(X, net);

        %back prop - CHECK
        delta2 = (y - V2);
        z2 = theta2'*delta2;
        z2 = z2(2:end,:);
        u1 = theta1*V0;
        delta1 = sigmoidGrad(u1, beta) .* z2;

        theta1(:,1) = 0;
        theta2(:,1) = 0;
        w2 = (1/B)*(lr * delta2 * V1' - lr * lambda * theta2);
        w1 = (1/B)*(lr * delta1 * V0' - lr * lambda * theta1);
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
function [X_out, y_out] = shuffleData(X, y)
    idx = randperm(size(y,2)); %generate shuffled indexes
    X_out = X(:, idx); %shuffle columns (each column is a training example)
    y_out = y(:, idx);
end
end

