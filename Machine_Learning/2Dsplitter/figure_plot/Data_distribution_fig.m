clear;
close all;

%% min(T) data %%
Data_dist = csvread('Data_distribution(06_min).csv');
xr = Data_dist(1, :);
Treward = Data_dist(2, :);

%% Plot %%
% plot properties
label_Fontsize = 20;
tick_Fontsize = 15;

% reward
figure;
bar(xr, Treward./1000);
xlabel('min(T)', 'Fontsize', label_Fontsize);
ylabel('# of data (x1000)', 'Fontsize', label_Fontsize);
ax = gca;
ax.FontSize = tick_Fontsize;
ax.FontWeight = 'bold';

%% avg(T) data %%
Avg_dist = csvread('Data_distribution(06_avg).csv');
xr = Avg_dist(1, :);
Treward = Avg_dist(2, :);

%% Plot %%
% plot properties
label_Fontsize = 20;
tick_Fontsize = 15;

% reward
figure;
bar(xr, Treward./1000);
xlabel('Average Transmission', 'Fontsize', label_Fontsize);
ylabel('# of data (x1000)', 'Fontsize', label_Fontsize);
ax = gca;
ax.FontSize = tick_Fontsize;
ax.FontWeight = 'bold';