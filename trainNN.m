function parameters = trainNN(X, y)

%NB output parameters is of form:
%[output_layer_size, hidden layer size, beta, neural net layer weights]

learning_rate = 0.01; %learning rate
beta = 1; %beta
lambda = 1; %regularisation

%----------------------------------------------------------------------------
%------------------------ SET ARCHITECTURE PARAMETERS -----------------------
%----------------------------------------------------------------------------

input_layer_size = size(X, 1); %set input layer size to number of features
hidden_layer_size = 100;
output_layer_size = 2; %set output layer size to 2 (predict changes in x and y positions)
layer_sizes = [input_layer_size, hidden_layer_size, output_layer_size];

%Initialise layer 1 and layer 2 weights
init_theta1 = initLayerWeights(input_layer_size, hidden_layer_size, 0.1); %initialise layer 1 weights
init_theta2 = initLayerWeights(hidden_layer_size, output_layer_size, 0.1); %initialise layer 2 weights

initial_weights = [init_theta1(:) ; init_theta2(:)]; %flatten weights into 1D vector

network = [learning_rate, beta, lambda, layer_sizes, initial_weights];

%----------------------------------------------------------------------------
%------------------------ STOCHASTIC GRADIENT DESCENT -----------------------
%----------------------------------------------------------------------------
fprintf('Training model...')
B = 100; %batch size
Epochs = 50;
for epoch = 1:epochs
    for sample = 1:B:length(y) %sweep through all data points
        %perform update of parameters for each training item
        X_batch = X(:, sample:sample+B-1);
        y_batch = y(:, sample:sample+B-1);
        nn_parameters = sweep(X_batch, y_batch, network);
    end
end
fprintf('Done.\n')
%----------------------------------------------------------------------------
%-------------------------- CONCATENATE PARAMETERS --------------------------
%----------------------------------------------------------------------------
%output parameters
parameters = [output_layer_size, hidden_layer_size, b, nn_parameters']; %output vector containing all required parameters


%----------------------------------------------------------------------------
%--------------------------------- FUNCTIONS --------------------------------
%----------------------------------------------------------------------------

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

function W = initLayerWeights(n_in, n_out, e)
    %initialise layer weights between -e and +e
    W = zeros(n_out, 1 + n_in);
    W = rand(n_out, 1 + n_in) * 2 * e - e;
end

function weights_new = sweep(X, y, weights, layer_sizes, b, lr, lambda)
    %Perform sweep through network of single training data element
    %1. Forward propagate
    %2. Back propagate
    %3. Update weights and output
    
    [weights, layer_sizes, b, lr, lambda] = expandArchitecture(net);
    
    [V0, V1, V2] = forwardPass(X, weights, layer_sizes, b);
    [w1, w2] = backwardPass(X, y, weights, layer_sizes, lr, b, lambda);

    theta1 = theta1 + w1;
    theta2 = theta2 + w2;

    weights_new = [theta1(:); theta2(:)];
end

function [V0, V1, V2] = forwardPass(X, net)
    [lr, b, lambda, layer_sizes, weights] = expandArchitecture(net);
    [theta1, theta2] = reshapeTheta(weights, layer_sizes);
   
    m = size(X,2); %get number of samples 
    V0 = [ones(1,m); X]; %add bias term
    
    %Hidden layer = sigmoid units
    u1 = theta1*V0;
    V1 = sigmoid(u1, b);
    
    %Output layer = linear units
    V1 = [ones(1,m); V1]; %add bias term
    u2 = theta2*V1;
    V2 = u2;
end

function [w1, w2] = backwardPass(X, y, net)
    %get weights and fwd propagate
    [lr, b, lambda, layer_sizes, weights] = expandArchitecture(net);

    [theta1, theta2] = reshapeTheta(net);
    [V0, V1, V2] = forwardPass(X, weights, layer_sizes, b);
    
    %back prop - CHECK
    delta2 = (y - V2);
    z2 = delta2'*theta2;
    z2 = z2(2:end);
    delta1 = sigmoidGrad(u1, b) .* z2';

    %calc weight updates w2, w1
    w2 = (lr * delta2 .* V1') - (2 * lr * lambda * theta2);
    w1 = (lr * delta1 .* V0') - (2 * lr * lambda * theta1);
end

function [theta1, theta2] = reshapeTheta(net)
    [~, ~, ~, layer_sizes, weights] = expandArchitecture(net);
    input_size = layer_size(1);
    hidden_size = layer_size(2);
    output_size = layer_size(3);
    
    %reshape layer weights into matrices
    theta1 = weights(1:hidden_size * (input_size + 1));
    theta1 = reshape(theta1, hidden_size, (input_size + 1));

    theta2 = weights((1 + (hidden_size * (input_size + 1))):end);
    theta2 = reshape(theta2, output_size, (hidden_size + 1));
end

function [lr, b, reg, LS, W] = expandArchitecture(net)
   lr = net(1);
   b = net(2);
   reg = net(3); %lambda
   LS = net(4:6); %layer sizes
   W = net(7:end); %weights
end

end

