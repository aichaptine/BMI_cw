function feature_vector = splice_data(data, B)
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
end

