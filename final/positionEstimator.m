function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %      SVM CLASSIFIER PREDICTION       %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        


% if (modelParameters{end} ~= 0) %i.e. if already predicted, get angle (so we dont have to predict every time)
%     predAngle = modelParameters{end};
% else
    %perform SVM to find predAngle
    %N.B SVM parameters stored in modelParameters{end-1}
    %update modelParameters{end} to predAngle
    %Using test_data, predict angle
    test_data_processed = preprocessPlanningData(test_data);
    out = zeros(28,size(test_data_processed, 2));
    models = modelParameters{end-1};
    for model=1:length(models)
        out(model,:) = svmPredict(models(model), test_data_processed');
    end
    predictions = convert_to_RA(out);
    predAngle = count_occurrences(predictions); 
    modelParameters{end} = predAngle;
    predAngle
% end

function pred = svmPredict(model, X)
%SVMPREDICT returns a vector of predictions using a trained SVM model
%(svmTrain). 
%   pred = SVMPREDICT(model, X) returns a vector of predictions using a 
%   trained SVM model (svmTrain). X is a mxn matrix where there each 
%   example is a row. model is a svm model returned from svmTrain.
%   predictions pred is a m x 1 column of predictions of {0, 1} values.
%

% Check if we are getting a column vector, if so, then assume that we only
% need to do prediction for a single example
if (size(X, 2) == 1)
    % Examples should be in rows
    X = X';
end

% Dataset 
m = size(X, 1);
p = zeros(m, 1);
pred = zeros(m, 1);

if strcmp(func2str(model.kernelFunction), 'linearKernel')
    % We can use the weights and bias directly if working with the 
    % linear kernel
    p = X * model.w + model.b;
elseif contains(func2str(model.kernelFunction), 'gaussianKernel')
    % Vectorized RBF Kernel
    % This is equivalent to computing the kernel on every pair of examples
    X1 = sum(X.^2, 2);
    X2 = sum(model.X.^2, 2)';
    K = bsxfun(@plus, X1, bsxfun(@plus, X2, - 2 * X * model.X'));
    K = model.kernelFunction(1, 0) .^ K;
    K = bsxfun(@times, model.y', K);
    K = bsxfun(@times, model.alphas', K);
    p = sum(K, 2);
else
    % Other Non-linear kernel
    for i = 1:m
        prediction = 0;
        for j = 1:size(model.X, 1)
            prediction = prediction + ...
                model.alphas(j) * model.y(j) * ...
                model.kernelFunction(X(i,:)', model.X(j,:)');
        end
        p(i) = prediction + model.b;
    end
end

% Convert predictions into 0 / 1
pred(p >= 0) =  1;
pred(p <  0) =  0;

end
function out_updated = convert_to_RA(out)

out_updated = zeros(size(out));
a = zeros(1,size(out,2));
m = 0;

    for i=1:7
        for j= i+1:8
            m = m+1;
            a(out(m,:)==1) = i;
            a(out(m,:)==0) = j;
            out_updated(m,:) = a;
        end
    end
    
end
function predicted_angles = count_occurrences(out)
    ysvm = zeros(1,8);
    predicted_angles = zeros(1,size(out,2));
    
    for i=1:size(out,2)
        v = out(:,i);
        for j=1:length(ysvm)
            ysvm(j) = sum(v==j);
        end
        [argvalue, argmax] = max(ysvm);
        predicted_angles(i) = argmax;
    end
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

if (size(test_data.spikes, 2) == 320)
    currentPos = test_data.startHandPos + decodedDeltaPos;
else
    currentPos = test_data.decodedHandPos(:,end) + decodedDeltaPos;
end

x = currentPos(1);
y = currentPos(2);

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
    feature_vector = feature_vector';
end

end

