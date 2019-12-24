clear;
close all;

Corr_NN = csvread('Corr(NN).csv');
Corr_ResNet = csvread('Corr(ResNet).csv');

R_NN = corrcoef(Corr_NN(1, :), Corr_NN(2, :));
R_ResNet = corrcoef(Corr_ResNet(1, :), Corr_ResNet(2, :));

%% NN
figure;
scatter(Corr_NN(1, :), Corr_NN(2, :), 30, 'filled', 'ko');
hold on;
plot(0:0.05:0.5, 0:0.05:0.5, 'r', 'Linewidth', 2);
xlabel('Target', 'Fontsize', 20);
ylabel('Prediction', 'Fontsize', 20);
pbaspect([1 1 1]);
ylim([0 0.5]);
xlim([0 0.5]);
xticks([0 0.1 0.2 0.3 0.4 0.5]);
ax = gca;
ax.FontSize = 15;
ax.FontWeight = 'bold';

%% ResNet
figure;
scatter(Corr_ResNet(1, :), Corr_ResNet(2, :), 30, 'filled', 'ko');
hold on;
plot([0:0.05:0.5], [0:0.05:0.5], 'r', 'Linewidth', 2);
xlabel('Target', 'Fontsize', 20);
ylabel('Prediction', 'Fontsize', 20);
pbaspect([1 1 1]);
ylim([0 0.5]);
xlim([0 0.5]);
xticks([0 0.1 0.2 0.3 0.4 0.5]);
ax = gca;
ax.FontSize = 15;
ax.FontWeight = 'bold';
