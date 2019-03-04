%%% 1D Distributed Bragg Reflctor with Q learning %%%
clc; clear; close all;

%% LEARNING SETTINGS
learnRate = 0.9; % How is new value estimate weighted against the old (0-1). 1 means all new and is ok for no noise situations.
discount = 0.9; % When assessing the value of a state & action, how important is the value of the future states?
maxEpi = 20; % Each episode is starting with air layers
wQ = 1;
wMSL = -1;

%% DBR SETTINGS
% DBR
Ngrid = 200; % Number of grid. It is same with the actions will be taken in episode
dx = 10; % distance unit
eps0 = 1; % air permittivity
epsi = 12.25; % siliocn permittivity
Layer = zeros(1,Ngrid);

% wavelength
minlam = 0;
maxlam = 600;
dlam = 25;
lambda = minlam:dlam:maxlam;
tarlam = 300;
tarlam_index = find(lambda == tarlam);

%% Functions
% function R = calR(Layer,lambda,Ngrid,dx,epsi,eps0)
% calculate reflectivity (R)
%
% function reward = rewardFunc(R,lambda,tarlam_index)
% calculate reward by Q-factor and Maximum Side Level (MSL)

%% Generate a state list
Q = zeros(Ngrid,2);

for episodes = 1:maxEpi
    for ng = 1:Ngrid
        % select an action value i.e. 1 or 0
        % which has the maximum value of Q in it
        % if more than one actions has same value than select randomly from
        % them
        [val,index] = max(Q(ng,:));
        [xx,yy] = find(Q(ng,:) == val);
        if size(yy,1) > 1
            index = 1+round(rand*(size(yy,1)-1));
            action = yy(index,1);
        else
            action = index;
        end
        
        % action 1 = 0 (air), action 2 = 1 (dielectric)
        Layer(ng) = (action-1);
        
        % calculate reward
        R = calR(Layer,lambda,Ngrid,dx,epsi,eps0);
        [Qfac,MSL] = rewardFunc(R,lambda,tarlam_index);
        reward = wQ*Qfac*(1e-6) + wMSL*MSL;
        % update information in Q for later use
        Q(ng,action) = Q(ng,action) + learnRate*(reward+discount*max(Q(ng,:))-Q(ng,action));
    end
end

R = calR(Layer,lambda,Ngrid,dx,epsi,eps0);
plot(lambda,R);
[Qfac,MSL] = rewardFunc(R,lambda,tarlam_index);
