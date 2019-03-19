clear all ;
load('monkeydata_training.mat');

%% CALCULATE F & H Matrixes
% to read for understanding: https://pdfs.semanticscholar.org/7f57/12394d340d89a4e27da802c6eac26ce10ee4.pdf

%100 ms at end are removed, 300 ms at beginning are removed , and we
%consider each bin to be 1ms and the 20 ms before and after, need to find
%optimal T and substitute it with end
%loop F for each time step of 20 ms? and divide in training and test each
%80% and 20%
%maximum T is 975 and the minimum is 571, assume T=571
angle=1;
T=571;
lambda_f = 1;
lambda_h = 1;
for i=1:80 %number of trials
hand_positions = trial(i,angle).handPos(1:2,301:end-100);
spike_rates = trial(i,angle).spikes(:,301:end-120); %if the X for H is the original X
X = hand_positions(1:2,1:end-20);
X_shift = hand_positions(1:2, 21:end);
F(:,:,i)= X_shift*X'*inv(X*X' + lambda_f*eye(2));
H(:,:,i) = spike_rates*X'*inv((X*X' + lambda_h*eye(2)));
end
F_for_all=mean(F,3);
H_for_all=mean(H,3);
%need to calculate also Q and R
for i=1:80
hand_positions = trial(i,angle).handPos(1:2,301:end-100);
spike_rates = trial(i,angle).spikes(:,301:end-120); %if the X for H is the original X
X = hand_positions(1:2,1:end-20);
X_shift = hand_positions(1:2, 21:end);    
EF=F_for_all*X - X_shift;
EH=H_for_all*X - spike_rates;
Q(:,:,i)=(EF*(EF)')/((size(hand_positions,2))-2);
R(:,:,i)=(EH*(EH)')/((size(hand_positions,2))-2);
end
Q_for_all=mean(Q,3);
R_for_all=mean(R,3);

%I am stuck because i dont understand how to get P which is required in
%order to calculate the Kalman gain



