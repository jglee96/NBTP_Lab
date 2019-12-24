clear;
close all;

c = 299792458;
%% Splitter %%
Splitter_T = csvread('1_TMmode_result_T.csv');
Splitter_R = csvread('1_TMmode_result_R.csv');
Raw_T = csvread('rawSample_T.csv');
Raw_R = csvread('rawSample_R.csv');
% Splitter_T = csvread('rewardsample_T.csv');
% Splitter_R = csvread('rewardsample_R.csv');
% Transmission polt properties
T_Linewidth = 2;
T_color = 'b';
% Reflection polt properties
R_Linewidth = 2;
R_color = 'r';
%% Plot %%
% plot properties
label_Fontsize = 20;
tick_Fontsize = 15;

% linear
figure;
plot(Splitter_T(:,1)*10^6, Splitter_T(:,2), T_color, 'Linewidth', T_Linewidth);
hold on;
plot(Splitter_R(:,1)*10^6, Splitter_R(:,2), R_color, 'Linewidth', R_Linewidth);
hold on;
plot(Raw_T(:,1)*10^6, Raw_T(:,2), T_color, 'Linewidth', T_Linewidth);
hold on;
plot(Raw_R(:,1)*10^6, abs(Raw_R(:,2)), R_color, 'Linewidth', R_Linewidth);
xlabel('Wavelength [um]', 'Fontsize', label_Fontsize);
ylabel('Transmission', 'Fontsize', label_Fontsize);
xlim([275 325]);
xticks([275 285 295 305 315 325]);
ax = gca;
ax.FontSize = tick_Fontsize;
ax.FontWeight = 'bold';

% dB
figure;
plot(Splitter_T(:,1)*10^6, 10*log10(Splitter_T(:,2)), T_color, 'Linewidth', T_Linewidth);
hold on;
plot(Splitter_R(:,1)*10^6, 10*log10(abs(Splitter_R(:,2))), R_color, 'Linewidth', R_Linewidth);
hold on;
plot(Raw_T(:,1)*10^6, 10*log10(Raw_T(:,2)), strcat('--',T_color), 'Linewidth', T_Linewidth);
hold on;
plot(Raw_R(:,1)*10^6, 10*log10(abs(Raw_R(:,2))), strcat('--',R_color), 'Linewidth', R_Linewidth);
xlabel('Wavelength [um]', 'Fontsize', label_Fontsize);
ylabel('Transmission [dB]', 'Fontsize', label_Fontsize);
xlim([275 325]);
xticks([275 285 295 305 315 325])
ax = gca;
ax.FontSize = tick_Fontsize;
ax.FontWeight = 'bold';

[Splitter_avg, Splitter_min, Splitter_max, Splitter_uniform] = Splitter(Splitter_T);
[Raw_avg, Raw_min, Raw_max, Raw_uniform] = Splitter(Raw_T);

function [avgT, minT, maxT, uniform] = Splitter(x)
avgT = mean(10*log10(x(:,2)));
minT = min(10*log10(x(:,2)));
maxT = max(10*log10(x(:,2)));
uniform = maxT - minT;
end
