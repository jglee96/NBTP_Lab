clear;
close all;

%%
Data_dist = csvread('Data_distribution(07_min).csv');
xr = Data_dist(1, :);
Treward = Data_dist(2, :);

%% Plot %%
% plot properties
label_Fontsize = 20;
tick_Fontsize = 15;

% reward
figure(1);
bar(xr, Treward./1000);
xlabel('min(T)', 'Fontsize', label_Fontsize);
ylabel('# of data (x1000)', 'Fontsize', label_Fontsize);
ax = gca;
ax.FontSize = tick_Fontsize;
ax.FontWeight = 'bold';