function [x y, modelParameters] = positionEstimator(test_data, modelParameters)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %      SVM CLASSIFIER PREDICTION       %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
%Using test_data, predict angle


if (modelParameters{end}.reach_angle ~= 0) %i.e. if already predicted, get angle (so we dont have to predict every time)
    predAngle = modelParameters{end}.reach_angle;
else
    %perform SVM to find predAngle
    %N.B SVM parameters stored in modelParameters{end-1}
    %update modelParameters{end} to predAngle
end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %       NN REGRESSOR PREDICTION        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

selectedNet = modelParameters{predAngle}; %select NN for chosen reaching angle
feat_vec = preprocessMovementData(test_data, 3, 20); %take previous 3x20ms of data, construct feature vector

%Normalise test data with parameters from training data
mx = selectedNet.mx;
sx = selectedNet.sx;
my = selectedNet.my;
feat_vec = (feat_vec-mx)./sx;
feat_vec((sx==0), :) = []; %remove rows with 0 std as no variation so dont contribute

decodedDeltaPos = predictNet(feat_vec, selectedNet); %predict change in x,y for given feature vector

decodedDeltaPos = decodedDeltaPos + my; %undo mean-centre of pos data

currentPosX = modelParameters{end}.current_pos(1); %read current positions
currentPosY = modelParameters{end}.current_pos(2); %read current positions
x = currentPosX + decodedDeltaPos(1);
y = currentPosY + decodedDeltaPos(2);

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

function feature_vector = preprocessMovementData(data, B, L)
            
    %Take only data corresponding to movement
    movement_spikes = data.spikes(:, end-B*L+1:end);
    N = size(movement_spikes, 1);
%     spike_sample = movement_spikes(:, k+1-B*L : k);

    %calculate features (firing rates) for all 98 neurons for each bin
    for i = 1:B
        bin = movement_spikes(:, 1+(i-1)*L : i*L); %take sub section of input data
        spike_count = sum(bin, 2); %count number of spikes per spike-train
        fr = spike_count*1000/L; %divide by length of spike-train
        feature_vector(1+(i-1)*N:i*N) = fr;
    end
end
% function feature_vector = spliceData(data, B)
% %Takes in data segment length T ms (i.e. NxT) and
% %1. Splits data into B segments (i.e segment size: N x T/B)
% %2. Extracts firing rates from each segment
% %Outputs 1-D vector of length (N*B x 1):
% %[----fr_bin1----, ----fr_bin2----, ..., ----fr_binB----]
% %e.g. fr_bin1 contains firing rates for neurons in 1st bin
%     [N, T] = size(data);
%     feature_vector = zeros(N*B, 1);
%     L = T/B; %L = bin length
%     
%     for i = 1:B
%         bin = data(:, 1+(i-1)*L : i*L); %take sub section of input data
%         spike_count = sum(bin, 2); %count number of spikes per spike-train
%         feature_vector = spike_count*1000/L; %divide by length of spike-train
%         feature_vector(1+(i-1)*N:i*N) = fr;
%     end
% 
% end


function X = preprocessPlanningData(data)
%Takes in test_data
%1. Calc firing rates for first 300ms -> column of X

    %Take only data corresponding to planning (<300ms)
    planning_spikes = data.spikes(:,1:300); 
    T = 300; %gives length of cut sample

    %Calc firing rate for all neurons in 300ms interval
    spike_count = sum(planning_spikes, 2); %count number of spikes per spike-train
    firing_rates = spike_count*1000/T; %divide by length of spike-train
    X = firing_rates;
end

end

