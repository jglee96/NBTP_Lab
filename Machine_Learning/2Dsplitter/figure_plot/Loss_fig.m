clear;
close all;

Training_NN = csvread('Training_loss(NN).csv');
Test_NN = csvread('Test_loss(NN).csv');
Training_ResNet = csvread('Training_loss(ResNet).csv');
Test_ResNet = csvread('Test_loss(ResNet).csv');

NN_test_mean = mean(Test_NN);
ResNet_test_mean = mean(Test_ResNet);

lw = 3;
fs = 20;

%% NN loss plot
figure;
plot(Training_NN, 'k', 'Linewidth', lw);
hold on;
plot(Test_NN, 'b', 'Linewidth', lw);
xlabel('Step', 'Fontsize', fs);
ylabel('Loss', 'Fontsize', fs);
ylim([0 0.1]);
xlim([0 100]);
ax = gca;
ax.FontSize = 15;
ax.FontWeight = 'bold';
% set(gca, 'YScale', 'log')

figure;
plot(Training_NN, 'k', 'Linewidth', lw);
hold on;
plot(Test_NN, 'b', 'Linewidth', lw);
xlabel('Step', 'Fontsize', fs);
ylabel('Loss', 'Fontsize', fs);
ax = gca;
ax.FontSize = 15;
ax.FontWeight = 'bold';
% set(gca, 'YScale', 'log')

%% ResNet loss plot
figure;
plot(Training_ResNet, 'k', 'Linewidth', lw);
hold on;
plot(Test_ResNet, 'b', 'Linewidth', lw);
xlabel('Step', 'Fontsize', fs);
ylabel('Loss', 'Fontsize', fs);
ylim([0 0.1]);
xlim([0 100]);
ax = gca;
ax.FontSize = 15;
ax.FontWeight = 'bold';

figure;
plot(Training_ResNet, 'k', 'Linewidth', lw);
hold on;
plot(Test_ResNet, 'b', 'Linewidth', lw);
xlabel('Step', 'Fontsize', fs);
ylabel('Loss', 'Fontsize', fs);
ax = gca;
ax.FontSize = 15;
ax.FontWeight = 'bold';