% 201811 Lee Jonggeon THz TDS setup
clear; clc;
close all;

c0=299792458;
ds = 5e-6;

%% load signal & save data %%

signal = load(sprintf('t1.txt'));

s = signal(:,1);
dt = 2*ds/c0;
t = 0:dt:(length(s)-1)*dt;

df = 1/dt;
f = linspace(0,df,length(s));
signal_fft = fft(signal(:,2));
signal_amp = abs(signal_fft);
signal_phase = unwrap(angle(signal_fft));

%% Plot the original signal %%

figure(1);
plot(t*1e12, signal(:,2)*1e9, 'LineWidth', 2);

xlabel('Time [ ps ]');
ylabel('THz signal [nA]');
set(gca,'Fontsize',15);
xlim([0 66.78]);

%% Plot the FFT signal %%

figure(2);
plot(f*1e-12, signal_amp, 'LineWidth', 2);
xlabel('Frequency [ THz ]');
ylabel('Amplitude (a.u.)');
set(gca,'Fontsize',15);
xlim([0 2]);

figure(3);
plot(f*1e-12, signal_phase, 'LineWidth', 2);
xlabel('Frequency [ THz ]');
ylabel('Phase (rad)');
set(gca,'Fontsize',15);
xlim([0 2]);
% ylim([-pi pi]);