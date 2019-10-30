clear;
close all;

c = 299792458;
%% Splitter %%
Splitter_T = csvread('TMmode_result_T.csv');
Splitter_R = csvread('TMmode_result_R.csv');
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
figure(1);
plot(Splitter_T(:,1)*10^6, Splitter_T(:,2), T_color, 'Linewidth', T_Linewidth);
hold on;
plot(Splitter_R(:,1)*10^6, Splitter_R(:,2), R_color, 'Linewidth', R_Linewidth);
xlabel('Wavelength [um]', 'Fontsize', label_Fontsize);
ylabel('Transmission', 'Fontsize', label_Fontsize);
xlim([275 325]);
xticks([275 285 295 305 315 325]);
ax = gca;
ax.FontSize = tick_Fontsize;

% dB
figure(2);
plot(Splitter_T(:,1)*10^6, 10*log10(Splitter_T(:,2)), T_color, 'Linewidth', T_Linewidth);
hold on;
plot(Splitter_R(:,1)*10^6, 10*log10(abs(Splitter_R(:,2))), R_color, 'Linewidth', R_Linewidth);
xlabel('Wavelength [um]', 'Fontsize', label_Fontsize);
ylabel('Insertion loss [dB]', 'Fontsize', label_Fontsize);
xlim([275 325]);
xticks([275 285 295 305 315 325])
ax = gca;
ax.FontSize = tick_Fontsize;