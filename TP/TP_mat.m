clear all;
close all;

%% load signal & save data %%

signal = load(sprintf('measured_signal.txt'));

t = signal(:,1);
signal(:,1) = [];

d = 737.5e-6;
c = 299792458;

%% Plot the original signal %%

figure(1);
plot(t, signal*10^9);

xlabel('Time [ps]');
ylabel('THz signal [nA]');
legend('Reference','Sample');
set(gca,'Fontsize',15);
xlim([0 90]);

%% Delete Extra(2nd,3rd and ...) signal %%

for j=1:length(t)
    if t(j)>40
        signal(j,2) = 0;
    end
end

%% Fast Fourier Transforms %%

signal_fft = fft(signal); % FFT signal
signal_amp = abs(signal_fft);
signal_ph(:,1) = phase(signal_fft(:,1));
signal_ph(:,2) = phase(signal_fft(:,2));

f = ((0:length(t)-1)*30/length(t))'; % time to frequency [THz]
k_0 = 2*pi*f*(10^12)./c;

%% Plot the incidnet and transmitted THz field sepctra %%

figure(2);
plot(f,signal_amp);

xlabel('Frequency [THz]');
ylabel('Amplitude');
legend('Reference','Sample');
set(gca,'Fontsize',15);
xlim([0 5]);

%% Plot the transmission coefficient %%

figure(3);
T_ex = signal_amp(:,2).*exp(-1i.*k_0.*d)./signal_amp(:,1);
plot(f, abs(T_ex));

xlabel('Frequency [THz]');
ylabel('Transmission');
xlim([0 5]);
set(gca,'Fontsize',15);

%% Calculate unknown index %%

n_ini = 1;

n(1:length(f)) = n_ini;
n_old = n_ini;
error = 1;

for w=2:length(f)
    n(w) = n(w-1);
    while abs(error)>1e-10 && n(w)+1~=0
        n_old = n(w);        
        
        T12 = (2*1)/(n(w)+1);
        T23 = (2*n(w))/(n(w)+1);
        
        T_th = T12*T23*exp(-1i*k_0(w)*n(w)*d);
        
        Delta = T_th - T_ex(w);
        %Delta_diff = (4*exp(-d*k_0(w)*1i))/(n(w) + 1)^2 - (8*n(w)*exp(-d*k_0(w)*1i))/(n(w) + 1)^3;
        Delta_diff = (4*exp(-d*k_0(w)*n(w)*1i))/(n(w) + 1)^2 - (8*n(w)*exp(-d*k_0(w)*n(w)*1i))/(n(w) + 1)^3 - (d*k_0(w)*n(w)*exp(-d*k_0(w)*n(w)*1i)*4i)/(n(w) + 1)^2;
        
        n(w) = n(w)-Delta/Delta_diff;
        
        error = abs((n(w)-n_old)/n(w));
    end
    error = 1;
end

%% Plot the transmission coefficient %%

figure(4);
plot(f, real(n));
hold on;
plot(f, imag(n));

xlabel('Frequency [THz]');
ylabel('index');
xlim([0 2]);



