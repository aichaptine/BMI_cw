function [X, y, id] = preprocess_movement_data(data, B, L)
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
    %B = 3; %set #bins
    %L = 20; %bin length
    
    y = [];
    X = [];
    id = [];
    
    for i = 1:N_trials
        for j = 1:N_angles %access each cell
            
            %Take only data corresponding to movement
            movement_spikes = data(i,j).spikes(:,300:end); %300 or 301?
            movement_handPos = data(i,j).handPos(:,300:end);
            T = size(movement_spikes, 2); %gives length of cut sample
            
            %Pass along spike-trains, take multi-bin segments (length B*L). Then get
            %firing rates for each neuron in each bin (this will give 98*B firing rates i.e. a vector length 98*B. 
            %Then place these firing rates in X, change in handPos in y and trialId in id
            for k = B*L:L:T %need to start from B*L e.g. start at 60ms (if 3bins of size 20ms) since we need previous 60 values
                spike_sample = movement_spikes(:, k+1-B*L : k);
                
                %calculate features (firing rates) for all 98 neurons for each bin
                feature_sample = splice_data(spike_sample, B);
                
                %calculate change in x,y position over 20ms (L) interval
                delta_handPos_sample = movement_handPos(1:2, k) - movement_handPos(1:2, k+1-L); 
                
                %append above to the output matrices
                X = [X, feature_sample];
                y = [y, delta_handPos_sample];
                id = [id, data(i,j).trialId];
            end
        end
    end
end
