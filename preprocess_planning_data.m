function [X, y, id] = preprocess_planning_data(data)
%Takes in 100x8 data structure, and for each trial
%1. Calc firing rates for first 300ms -> column of X
%2. Takes reaching angle for trial -> y
%3. Takes trial id -> id

    [N_trials, N_angles] = size(data);
    
    y = [];
    X = [];
    id = [];

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
                id = [id, data(i,j).trialId];
            end
        end
end

