clear all ;
load('monkeydata_training.mat');

%% CALCULATE F & H Matrixes
% to read for understanding: https://pdfs.semanticscholar.org/7f57/12394d340d89a4e27da802c6eac26ce10ee4.pdf
% to read for P : https://www.chegg.com/homework-help/questions-and-answers/derive-hand-procedure-kalman-filter-based-following-assumptions-write-derivations-paper-ha-q31439062
%100 ms at end are removed, 300 ms at beginning are removed , and we
%consider each bin to be 1ms and the 20 ms before and after, need to find
%optimal T and substitute it with end
%loop F for each time step of 20 ms? and divide in training and test each
%80% and 20%
%maximum T is 975 and the minimum is 571, assume T=571
%might need to initialize P as expected value of the predicted x and the
%actual one at t=0
clear all;
%close all;
load('monkeydata_training.mat');
angle=5;
lambda_f = 125;
lambda_h = 125;
%T = 571;
figure;
hold on;
for angle=1:8
for i=1:80 %number of trials
    hand_positions = trial(i,angle).handPos(1:2,301:end-100);
    spike_rates = trial(i,angle).spikes(:,301:end-20-100); %if the X for H is the original X
    X = hand_positions(1:2,1:end-20);
    X_shift = hand_positions(1:2, 21:end);
    F(:,:,i)= X_shift*X'*inv(X*X' + lambda_f*eye(2));
    H(:,:,i) = spike_rates*X'*inv((X*X' + lambda_h*eye(2)));
end
F = mean(F,3);
H = mean(H,3);

%need to calculate also Q and R
for i=1:80
    hand_positions = trial(i,angle).handPos(1:2,301:end-100);
    spike_rates = trial(i,angle).spikes(:,301:end-20-100); %if the X for H is the original X
    X = hand_positions(1:2,1:end-20);
    X_shift = hand_positions(1:2, 21:end);    
    EF = F*X - X_shift;
    EH = H*X - spike_rates;
    Q(:,:,i)=(EF*(EF)')/((size(hand_positions,2))-2);
    R(:,:,i)=(EH*(EH)')/((size(hand_positions,2))-2);
end
Q = mean(Q,3);
R = mean(R,3);


%NOW lets try do predict the test trials
%prime is t' 


for tr = 81:100
    trial_length = length(trial(tr,angle).handPos) - (300+100);
    spike_rates = trial(tr,angle).spikes(:,301:end-100);
    predicted_handpos = zeros(2,trial_length);
    predicted_handpos(:,1) = trial(tr,angle).handPos(1:2,1);
    for timestep = 1:trial_length
        if(timestep == 1)
            P_previous = cov(0,0);
            %predicted_handpos(1,1),predicted_handpos(2,1));
        else
            P_prime = F*P_previous*F' + Q;   %need to fix matrix itself cause wont work like this
            S = H * P_prime * H' + R;
            K_gain = P_prime * H' * inv(nearestSPD(S));
            X_prime = F * predicted_handpos(:,timestep - 1);
            Z = H*X_prime;
            predicted_handpos(:,timestep) = X_prime + K_gain*(spike_rates(:,timestep)-Z);
            P_previous = (eye(2) - K_gain*H)*P_prime;
            warning off;
        end
    end
    
    plot(predicted_handpos(1,:),predicted_handpos(2,:),'b');
    plot(trial(tr,angle).handPos(1,301:end-100),trial(tr,angle).handPos(2,301:end-100),'r'); 
    legend({'predicted','real'});
    meanerror = mean(mean(predicted_handpos - trial(tr,angle).handPos(1:2,301:end-100),2),1);
    RMSE(tr) = sqrt(meanerror^2);
    display("Angle: " + num2str(angle));
end
end
RMSE_all = mean(RMSE);
%initial and last bin set to 20 can manipulate as u wish 




