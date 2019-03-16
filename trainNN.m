function parameters = trainNN(X, y)

%NB output parameters is of form:
%[output_layer_size, hidden layer size, beta, neural net layer weights]

lr = 0.01; %learning rate
b = 1; %beta
lambda = 1; %regularisation

%----------------------------------------------------------------------------
%------------------------ SET ARCHITECTURE PARAMETERS -----------------------
%----------------------------------------------------------------------------

input_layer_size = size(X, 1); %set input layer size to number of features
hidden_layer_size = 100;
output_layer_size = 2; %set output layer size to 2 (predict changes in x and y positions)

%Initialise layer 1 and layer 2 weights
init_theta1 = initLayerWeights(input_layer_size, hidden_layer_size, 0.1); %initialise layer 1 weights
init_theta2 = initLayerWeights(hidden_layer_size, output_layer_size, 0.1); %initialise layer 2 weights

initial_parameters = [init_theta1(:) ; init_theta2(:)]; %flatten weights into 1D vector
nn_parameters = initial_parameters;

%----------------------------------------------------------------------------
%------------------------ STOCHASTIC GRADIENT DESCENT -----------------------
%----------------------------------------------------------------------------
fprintf('Training model...')
for sample = 1:length(y) %sweep through all data points
    %perform update of parameters for each training item
    nn_parameters = sweep(X(:,sample), y(sample), nn_parameters, input_layer_size, ...
        hidden_layer_size, output_layer_size, b, lr, lambda);
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

function parameters_updated = sweep(X, y, parameters, input_layer_size, hidden_layer_size, output_layer_size, b, lr, lambda)
    %Perform sweep through network of single training data element
    %1. Forward propagate
    %2. Back propagate
    %3. Update weights and output
    
    m = size(X,2); %get number of samples 
    
    %reshape layer weights into matrices
    theta1 = parameters(1:hidden_layer_size * (input_layer_size + 1));
    theta1 = reshape(theta1, hidden_layer_size, (input_layer_size + 1));

    theta2 = parameters((1 + (hidden_layer_size * (input_layer_size + 1))):end);
    theta2 = reshape(theta2, output_layer_size, (hidden_layer_size + 1));

    %fwd propagate
    %Hidden layer = sigmoid units
    V0 = [ones(1,m); X]; %add bias term
    u1 = theta1*V0;
    V1 = sigmoid(u1, b);
    
    %Output layer = linear units
    V1 = [ones(1,m); V1]; %add bias term
    u2 = theta2*V1;
    V2 = u2;
    
    %back prop
    delta2 = (y - V2);
    z2 = delta2'*theta2;
    z2 = z2(2:end);
    delta1 = sigmoidGrad(u1, b) .* z2';

    weight_update2 = lr * delta2 .* V1';
    weight_update1 = lr * delta1 .* V0';

    %regularisation term
    reg2 = 2 * lr * lambda * theta2;
    reg1 = 2 * lr * lambda * theta1;
    
    theta2 = theta2 + weight_update2 - reg2;
    theta1 = theta1 + weight_update1 - reg1;
    parameters_updated = [theta1(:); theta2(:)];
end

end

