%%% 1D Distributed Bragg Reflctor with Q learning %%%
clc; clear; close all;

%% LEARNING SETTINGS
learnRate = 0.9; % How is new value estimate weighted against the old (0-1). 1 means all new and is ok for no noise situations.
discount = 0.9; % When assessing the value of a state & action, how important is the value of the future states?
maxEpi = 100; % Each episode is starting with air layers
rQ = 1e0; % Q-factor scaling factor for reward
rMSL = 1; % MSL scaling factor for reward

%% DBR SETTINGS
% DBR
Ngrid = 200; % Number of grid. It is same with the actions will be taken in episode
dx = 25; % distance unit
eps0 = 1; % air permittivity
epsi = 12.25; % siliocn permittivity
Layer = zeros(1,Ngrid); % state
action = [0 1];

% wavelength
minlam = 25;
maxlam = 600;
dlam = 25;
lambda = minlam:dlam:maxlam;
tarlam = 300;
tarlam_idx = find(lambda == tarlam);

%% Functions
% function R = calR(Layer,lambda,Ngrid,dx,epsi,eps0)
% calculate reflectivity (R)
%
% function reward = rewardFunc(R,lambda,tarlam_index)
% calculate reward by Q-factor and Maximum Side Level (MSL)

%% Start Learning!!
Q = zeros(Ngrid,2);

for episodes = 1:maxEpi
    for state_idx = 1:Ngrid
        % select an action value i.e. 1 or 0
        % which has the maximum value of Q in it
        % if more than one actions has same value than select randomly from
        % them
        [val,index] = max(Q(state_idx,:));
        [xx,yy] = find(Q(state_idx,:) == val);
        if size(yy,2) > 1
            index = 1+round(rand);
            action_idx = yy(1,index);
        else
            action_idx = index;
        end
        
        % action 1 = 0 (air), action 2 = 1 (dielectric)
        Layer(state_idx) = action(action_idx); % permitivitty will apply on calR function
        
        % calculate reward
        R = calR(Layer,lambda,Ngrid,dx,epsi,eps0);
        [Qfac,MSL] = rewardFunc(R,lambda,tarlam_idx);
        tar_int = R(tarlam_idx);
        reward = (rQ*Qfac)/(rMSL*MSL);
        disp(['State: ',num2str(state_idx),'; Qfac: ',num2str(Qfac),'; MSL: ',num2str(MSL),'; reward: ',num2str(reward)]);
        if state_idx ~= Ngrid
            % update information in Q for later use
            Q(state_idx,action_idx) = Q(state_idx,action_idx) + learnRate*(reward+discount*max(Q(state_idx+1,:))-Q(state_idx,action_idx)); % state_idx is next state_idx
        end
    end
    figure(1);
    imagesc(Layer);
    colormap(gray);
    disp(['Episode: ',num2str(episodes),'; Qfac: ',num2str(Qfac),'; MSL: ',num2str(MSL),'; reward: ',num2str(reward)]);
    drawnow;
    
    figure(2);
    plot(lambda,R);
    drawnow;
end

R = calR(Layer,lambda,Ngrid,dx,epsi,eps0);
plot(lambda,R);
[Qfac,MSL] = rewardFunc(R,lambda,tarlam_idx);
