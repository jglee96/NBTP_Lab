clear all;
close all;
clc

%% load signal & save data %%

L=30000;
signal = zeros(L,2);
signal_temp = load('measured_signal.txt');

t = signal_temp(:,1);
signal_temp(:,1) = [];

signal(1:length(signal_temp),1) = signal_temp(:,1);
signal(1:length(signal_temp),2) = signal_temp(:,2);

d = 737.5*1e-6;
c = 299792458;

%% Plot the original signal %%

figure(1);
plot(t, signal_temp*10^9,'linewidth',2);

xlabel('Time [ps]');
ylabel('THz signal [nA]');
legend('Reference','Sample');
set(gca,'Fontsize',15);
xlim([0 90]);

%% Delete Extra(2nd,3rd and ...) signal %%

for j=1:length(t)
    if t(j)>40
        signal(j,:) = 0;
    end
end

%% Fast Fourier Transforms %%

signal_fft = fft(signal); % FFT signal
signal_amp = abs(signal_fft);
signal_ph(:,1) = phase(signal_fft(:,1));
signal_ph(:,2) = phase(signal_fft(:,2));

f_span = 1/(t(2)-t(1));
f = linspace(0,f_span,L+1)'; % time to frequency [THz]
f(L+1)=[];
k_0 = 2*pi*f*(10^12)./c;

%% Plot the incidnet and transmitted THz field sepctra %%

figure(2);
plot(f,signal_amp,'linewidth',2);
% figure(21);
% plot(f,signal_ph);

xlabel('Frequency [THz]');
ylabel('Amplitude');
legend('Reference','Sample');
set(gca,'Fontsize',15);
xlim([0 5]);

%% Plot the transmission coefficient %%

figure(3);
T_ex = signal_fft(:,2).*exp(-d.*k_0.*1i)./signal_fft(:,1);
plot(f, abs(T_ex),'linewidth',3);

xlabel('Frequency [THz]');
ylabel('Transmission');
xlim([0 2.2]);
%xlim([0 5]);
ylim([0 1.0]);
set(gca,'Fontsize',15);

%% Calculate unknown index %%

n_ini = 3.4-0.1i;
n(1:length(f),1) = n_ini;
T12 = zeros(length(f),1);
T23 = zeros(length(f),1);
T_th = zeros(length(f),1);
error = 1;
epsilon = 1e-10;
point = 2;

for w=point:length(f)
    n(w) = n(w-1);
    while abs(error)>epsilon
        n_old = n(w);
        
        T12(w) = (2*1)/(n(w)+1);
        T23(w) = (2*n(w))/(n(w)+1);
        
        T_th(w) = T12(w)*T23(w)*exp(-1i*k_0(w)*n(w)*d);
        
        df = T_th(w) - T_ex(w);
        df_diff = (4*exp(-d*k_0(w)*n(w)*1i))/(n(w) + 1)^2 - (8*n(w)*exp(-d*k_0(w)*n(w)*1i))/(n(w) + 1)^3 - (d*k_0(w)*n(w)*exp(-d*k_0(w)*n(w)*1i)*4i)/(n(w) + 1)^2;
                
        n(w) = n(w)-df/df_diff;      
        
        if isnan(n(w))
            n(w) = 0.5;
        end
        error = abs((n(w)-n_old)./n(w));
    end
    error = 1;
  
end
%% Plot the transmission coefficient %%

figure(4);
plot(f, real(n),'linewidth',3);
hold on;
plot(f, imag(n),'linewidth',3);

xlabel('Frequency [THz]');
ylabel('Index');
legend('real(n)', 'imag(n)');
set(gca,'Fontsize',15);
xlim([0.01 2]);
%ylim([-2 10]);





