clear all
close all
clc
% format long

FS='fontsize';
fs=20;

S = load('measured_signal.txt');
Ei = S(:,2);
Et = S(:,3);

NNN = 2602; % sampling numbers

% sum_Ei = sum(Ei)/2602;
% sum_Et = sum(Et)/2602;

d = 737.5 * 10^-6; % thickness

Ei_nano = Ei * 10^9;
Et_nano = Et * 10^9;

Ei_w = fft(Ei, NNN);
Et_w = fft(Et, NNN);

Ts = S(:,1);

Dt = Ts(2)-Ts(1);
Fs = 1/Dt;
% Df = Fs/size(S,1);
Df = Fs / NNN;


figure(1);
plot(Ts,Ei_nano,'-k', Ts, Et_nano, '-r','LineWidth',2)
axis([10 80 -50 50])
h1 = legend('Reference', 'Transmission');
xlabel('TIme (ps)', FS, fs)
ylabel('THz Signal (nA)', FS, fs)
grid on;
title('Time Domain Signal', FS, fs)
set(gca,FS,16)
set(h1,FS,14)

figure(1);
plot(Ts, Et_nano, '-r','LineWidth',2)
axis([10 80 -50 50])
h1 = legend('Transmission');
xlabel('TIme (ps)', FS, fs)
ylabel('THz Signal (nA)', FS, fs)
grid on;
title('Time Domain Signal', FS, fs)
set(gca,FS,16)
set(h1,FS,14)

Ei_ww = Ei_w(1:length(Ei_w)/2+1); 
Et_ww = Et_w(1:length(Et_w)/2+1); 

freq = 0:Df:Fs/2;

n = length(freq);

figure(2);
% subplot(3, 1, 1), 
plot(freq, abs(Ei_ww), '-k', freq, abs(Et_ww), '-r','LineWidth',2)
h2 = legend('Reference', 'Transmission');
xlabel('Frequency (THz)', FS, fs)
ylabel('Amplitude (a. u.)', FS, fs) 
grid on;
title('Frequency Domain Signal', FS, fs)
xlim([0 3])
ylim([0 1E-6])
set(gca,FS,16)
set(h2,FS,14)

figure(3);
% subplot(3, 1, 2), 
plot(freq, phase(Ei_ww), '-k', freq, phase(Et_ww), '-r','LineWidth',2)    
h3 = legend('Reference', 'Transmission');
xlabel('Frequency (THz)', FS, fs)
ylabel('Phase (degree)', FS, fs)
title('Frequency Domain Signal', FS, fs) 
grid on;
xlim([0 3])
set(gca,FS,16)
set(h3,FS,14)

figure(4);
% subplot(3, 1, 3), 
plot(freq, phase(Ei_ww)-phase(Et_ww),'LineWidth',2)
xlabel('Frequency (THz)', FS, fs)
ylabel('Phase (degree)', FS, fs)
title('Phase difference', FS, fs)
grid on;
xlim([0 3])
set(gca,FS,16)


c = 3*10^8;
lambda = c./(freq.*10^12); 
k_air = 2*pi*(freq*10^12)/c;

T_coeff = (Et_ww./Ei_ww).*exp(-j.*k_air'*d);

figure(5);
plot(freq, abs(T_coeff),'LineWidth',2)
xlabel('Frequency (THz)', FS, fs)
title('Transmission coefficient', FS, fs) 
grid on;
set(gca,FS,16)
xlim([0 3])
ylim([0 1])

figure(6);
plot(freq, phase(T_coeff),'LineWidth',2)
xlabel('Frequency (THz)', FS, fs)
ylabel('Phase (degree))', FS, fs) 
grid on;
set(gca,FS,16)
xlim([0 3])

E_iii = [Ei(1: 1111); zeros(1491,1)];
Eiii_w = fft(E_iii, NNN);
Eiii_ww = Eiii_w(1:length(Eiii_w)/2+1); 

E_ttt = [Et(1: 1201); zeros(1401,1)];
Ettt_w = fft(E_ttt, NNN);
Ettt_ww = Ettt_w(1:length(Ettt_w)/2+1); 

E_ttt2 = [zeros(1351,1); Et(1352: 1651); zeros(951,1)];
Ettt_w2 = fft(E_ttt2, NNN);
Ettt_ww2 = Ettt_w2(1:length(Ettt_w2)/2+1); 

E_ttt3 = [zeros(1801,1); Et(1802: 2602)];
Ettt_w3 = fft(E_ttt3, NNN);
Ettt_ww3 = Ettt_w3(1:length(Ettt_w3)/2+1); 


Eiii_nano = E_iii * 10^9;
Ettt_nano = E_ttt * 10^9;

figure(7);
plot(Ts,Eiii_nano,'-k', Ts, Ettt_nano, '-r','LineWidth',2)
axis([10 80 -50 50])
h7 = legend('Reference', 'Transmission');
xlabel('TIme (ps)', FS, fs)
ylabel('THz Signal (nA)', FS, fs)
grid on;
title('Time Domain Signal', FS, fs)
set(gca,FS,16)
set(h7,FS,14)

figure(8);
plot(freq, abs(Eiii_ww), '-k', freq, abs(Ettt_ww), '-r','LineWidth',2)
xlabel('Frequency (THz)', FS, fs)
ylabel('Amplitude (a. u.)', FS, fs) 
h8 = legend('Reference', 'Transmission');
grid on;
title('Frequency Domain Signal', FS, fs)
set(gca,FS,16)
set(h8,FS,14)

xlim([0 3])
ylim([0 1E-6])


T_coeff2 = (Ettt_ww./Eiii_ww).*exp(-j.*k_air'*d);

figure(9);
plot(freq, abs(T_coeff2),'LineWidth',2)
xlabel('Frequency (THz)', FS, fs)
title('Transmission coefficient', FS, fs) 
grid on;
set(gca,FS,16)
xlim([0 3])
ylim([0 1])

imax = 100;
Freq = freq * 10^12;
Freq = Freq';
Middle = 1;

for s = Middle:1302
    x= 3.4;
    P = 2 * pi * Freq(s) * d / c;
 
    for k = 1 : imax
        Fun =  (4*x*exp(-j*P*x))/(((x+1)^2)-((x-1)^2)*exp(-2*j*P*x))-T_coeff(s);
        Funder = ((4*exp(-j*P*x)+4*x*(-j*P)*exp(-j*P*x))*(((x+1)^2)-((x-1)^2)*exp(-2*j*P*x))-(4*x*exp(-j*P*x))*(2*(x+1)-(2*(x-1)*exp(-2*j*P*x)-2*j*P*((x-1)^2)*exp(-2*j*P*x))))/(((x+1)^2)-((x-1)^2)*exp(-2*j*P*x))^2;

        x = x - Fun/Funder; 
    end
    Index(s) = x;
end
   
for s = Middle:-1:1
    x= 3.4;
    P = 2 * pi * Freq(s) * d / c;
 
    for k = 1 : imax
        Fun =  (4*x*exp(-j*P*x))/(((x+1)^2)-((x-1)^2)*exp(-2*j*P*x))-T_coeff(s);
        Funder = ((4*exp(-j*P*x)+4*x*(-j*P)*exp(-j*P*x))*(((x+1)^2)-((x-1)^2)*exp(-2*j*P*x))-(4*x*exp(-j*P*x))*(2*(x+1)-(2*(x-1)*exp(-2*j*P*x)-2*j*P*((x-1)^2)*exp(-2*j*P*x))))/(((x+1)^2)-((x-1)^2)*exp(-2*j*P*x))^2;
  
        x = x - Fun/Funder; 
    end
    Index1(s) = x;
end

Index(1:Middle) = Index1(1:Middle);
freq_R = freq/n : freq/n : 1302*freq/n ;
freq_R = freq_R';

Index_re = real(Index);
Index_im = imag(Index);

figure(10);
plot(Ts, Ettt_nano, 'r', 'LineWidth',2)
axis([10 80 -50 50])
xlabel('TIme (ps)', FS, fs)
ylabel('THz Signal (nA)', FS, fs)
grid on;
title('Time Domain Transmission Signal', FS, fs)
set(gca,FS,16)


figure(11);
plot(freq, Index_re, 'LineWidth',2)
xlabel('Frequency (THz)', FS, fs)
ylabel('n_{real}', FS, fs)
grid on;
title('Refractive index (real)', FS, fs)
set(gca,FS,16)
axis([0, 30, 2.5, 4.0])
% xlim([0 3])

figure(12);
plot(freq, Index_im, 'LineWidth',2)
xlabel('Frequency (THz)', FS, fs)
ylabel('n_{imag}', FS, fs)
grid on;
title('Refractive index (imag)', FS, fs)
set(gca,FS,16)
axis([0,30,-1,0.5])
% xlim([0 3])

figure(13);
plot(freq, abs(Ettt_ww), '-k', freq, abs(Ettt_ww2), '-r',freq, abs(Ettt_ww3), '-b', 'LineWidth',2)
xlabel('Frequency (THz)', FS, fs)
ylabel('Amplitude (a. u.)', FS, fs) 
h13 = legend('1st signal', '2nd signal', '3rd signal');
grid on;
title('Frequency Domain Signal', FS, fs)
set(gca,FS,16)
set(h13,FS,14)
xlim([0 3])
ylim([0 1E-6])

for s = Middle:1302
    x= 3.4;
    P = 2 * pi * Freq(s) * d / c;
 
    for k = 1 : imax
        Fun =  (4*x*exp(-j*P*x))/((x+1)^2)-T_coeff2(s);
        Funder = ((4*exp(-j*P*x)+4*x*(-j*P)*exp(-j*P*x))*((x+1)^2)-(8*x*(x+1)*exp(-j*P*x)))/(x+1)^4;
        
    %    F_n = 
    %    dF_n

        x = x - Fun/Funder; 
    end
    Index2(s) = x;
end
   
for s = Middle:-1:1
    x= 3.4;
    P = 2 * pi * Freq(s) * d / c;
 
    for k = 1 : imax
        Fun =  (4*x*exp(-j*P*x))/((x+1)^2)-T_coeff2(s);
        Funder = ((4*exp(-j*P*x)+4*x*(-j*P)*exp(-j*P*x))*((x+1)^2)-(8*x*(x+1)*exp(-j*P*x)))/(x+1)^4;
        
    %    F_n = 
    %    dF_n

        x = x - Fun/Funder; 
    end
    Index3(s) = x;
end

Index2(1:Middle) = Index3(1:Middle);

Index2_re = real(Index2);
Index2_im = imag(Index2);

figure(14);
plot(freq, Index2_re, 'LineWidth',2)
xlabel('Frequency (THz)', FS, fs)
ylabel('n_{real}', FS, fs)
grid on;
title('Refractive index (real)', FS, fs)
set(gca,FS,16)
axis([0, 2, 2.5, 4.0])
% xlim([0 3])

figure(15);
plot(freq, Index2_im, 'LineWidth',2)
xlabel('Frequency (THz)', FS, fs)
ylabel('n_{imag}', FS, fs)
grid on;
title('Refractive index (imag)', FS, fs)
set(gca,FS,16)
axis([0,2,-1,0.5])

figure(16);
plot(freq, phase(Ettt_ww), '-k', freq, phase(Ettt_ww2), '-r',freq, phase(Ettt_ww3), '-b', 'LineWidth',2)
xlabel('Frequency (THz)', FS, fs)
ylabel('Phase (degree)', FS, fs) 
h13 = legend('1st signal', '2nd signal', '3rd signal');
grid on;
set(gca,FS,16)
set(h13,FS,14)
xlim([0 3])

figure(17);
plot(freq, phase(Ettt_ww)-phase(Ettt_ww2), '-k', freq, phase(Ettt_ww)-phase(Ettt_ww3), '-r', 'LineWidth',2)
xlabel('Frequency (THz)', FS, fs)
ylabel('Phase (degree)', FS, fs) 
h13 = legend('1st and 2nd signal', '1st and 3rd signal');
grid on;
set(gca,FS,16)
set(h13,FS,14)
xlim([0 3])
