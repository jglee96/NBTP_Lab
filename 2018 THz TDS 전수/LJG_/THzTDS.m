% 201811 Lee Jonggeon THz TDS setup
clear; clc;
close all;

%% load signal & save data %%

signal = load(sprintf('t1.txt'));

t = signal(:,1);
signal(:,1) = [];

%% Plot the original signal %%

figure(1);
plot(t, signal*10^9);

xlabel('Time [ps]');
ylabel('THz signal [nA]');
set(gca,'Fontsize',15);
% xlim([0 90]);