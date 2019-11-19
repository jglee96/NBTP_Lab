clear;
close all;

c = 299792458;
%% Splitter %%
Grid_size = [9, 10, 11, 12, 13, 14, 15];
Accuracy = [100, 99, 89, 72, 41, 15, 1];
%% Plot %%
% plot properties
label_Fontsize = 20;
tick_Fontsize = 15;

% linear
figure(1);
plot(Grid_size, Accuracy, 'k', 'Linewidth', 4);
xlabel('\it n', 'Fontweight','bold', 'Fontsize', label_Fontsize);
ylabel('Accuracy [%]', 'Fontweight','bold', 'Fontsize', label_Fontsize);
ylim([0 110]);
ax = gca;
ax.FontSize = tick_Fontsize;
ax.FontWeight = 'bold';