function firing_rate = compute_rate(spike_trains)
% Takes in an NxT matrix containing spike-trains length T for N  different neural units
% Outputs N-dim vector containing firing rate for each neural unit
    [N, T] = size(spike_trains);
    firing_rate = zeros(N, 1);
    
    spike_count = sum(spike_trains, 2); %count number of spikes per spike-train
    firing_rate = spike_count*1000/T; %divide by length of spike-train
end
