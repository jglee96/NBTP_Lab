%%% 1D Distributed Bragg Reflctor with Q learning %%%
clc; clear; close all;

%% LEARNING SETTINGS
learnRate = 0.9; % How is new value estimate weighted against the old (0-1). 1 means all new and is ok for no noise situations.
discount = 0.9; % When assessing the value of a state & action, how important is the value of the future states?
maxEpi = 100; % Each episode is starting with air layers
nact = 4; % number of actions
rQ = 1e0; % Q-factor scaling factor for reward
rMSL = 1; % MSL scaling factor for reward

%% DBR SETTINGS
% DBR
Ngrid = 50; % Number of grid. It is same with the actions will be taken in episode
dx = 5; % distance unit
eps0 = 1; % air permittivity
epsi = 12.25; % siliocn permittivity
Layer = zeros(1,Ngrid); % state

% wavelength
minlam = 10;
maxlam = 600;
dlam = 10;
lambda = minlam:dlam:maxlam;
tarlam = 800;
tarlam_idx = find(lambda == tarlam);

%% Functions
% function R = calR(Layer,lambda,Ngrid,dx,epsi,eps0)
% calculate reflectivity (R)
%
% function reward = rewardFunc(R,lambda,tarlam_index)
% calculate reward by Q-factor and Maximum Side Level (MSL)

%% Start Learning!!

% define action value function
% state: each pixel can be 1 or 0. -> number of states = Ngrid*2
% action: chage state; 1:next state,opposite state; 2:next state,same
% state; -> number of actions = 2
Q = zeros(2*Ngrid,nact);
% set boundary conditions
Q(1,[1,2]) = -Inf;
Q(Ngrid+1,[1,2]) = -Inf;
Q(Ngrid,[3,4]) = -Inf;
Q(2*Ngrid,[3,4]) = -Inf;

for episodes = 1:maxEpi
    state_idx = 1;
    %prevreward = 0;
    while (state_idx ~= Ngrid) && (state_idx ~= 2*Ngrid)
        % select an action value i.e. 1,2
        % which has the maximum value of Q in it
        % if more than one actions has same value than select randomly from
        % them
        [val,index] = max(Q(state_idx,:));
        [xx,yy] = find(Q(state_idx,:) == val);
        if size(yy,2) > 1
            rng('shuffle');
            index = round(randi(size(yy,2)));
            action_idx = yy(index);
        else
            action_idx = index;
        end
        
        [Layer,newstate_idx] = DoActLayer(Layer,state_idx,action_idx);
        
        % calculate reward
        R = calR(Layer,lambda,Ngrid,dx,epsi,eps0);
        [Qfac,MSL] = rewardFunc(R,lambda,tarlam_idx);
        tar_int = R(tarlam_idx);
        reward = (rQ*Qfac)/(rMSL*MSL);
        %effreward = reward-prevreward;
        disp(['State: ',num2str(mod(state_idx,Ngrid)),'; Qfac: ',num2str(Qfac),'; MSL: ',num2str(MSL),'; reward: ',num2str(reward)]);
        
        Q(state_idx,action_idx) = Q(state_idx,action_idx) + learnRate*(reward+discount*max(Q(newstate_idx,:)+rand/(episodes+1))-Q(state_idx,action_idx)); % state_idx is next state_idx
        
        state_idx = newstate_idx;
        %prevreward = reward;
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
