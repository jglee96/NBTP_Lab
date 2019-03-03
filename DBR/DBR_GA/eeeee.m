clear all
close all
clc
format long

FS = 'FontSize'; fs1 = 16; fs2 = 18; fs3 = 20;
LS = 'linestyle'; NN ='none';
LN = 50;
LW = 'linewidth'; lw=1.5;
LV = 10;
%pol = input('TE : 1 , TM : 2');
pol = 1;
%nl  = input('Input the number of layer : ');
layer = 12;
c  = 3e8;
f = (150:1:3000)*1e12;
lambda = c./f;
%% Transfer Matrix TE

E_in = 1;
lambda_t = 800*1e-9;
n_air = 1; % air index
k_air = 2*pi./lambda; % air wave vector
d_air = n_air*lambda_t/4;

n_mat = 1.95%+1i*0.00525; % material index
k_mat = n_mat*2*pi./lambda; % material wave vector
d_mat = lambda_t/4/real(n_mat);

n_cav = 1; % air index
k_cav = 2*pi./lambda; % air wave vector
d_gain = 200e-6;
im = sqrt(-1);
kax = k_air; kmx = k_mat; kcx = k_cav;
x = (0:1:2500)*1e-6;

switch pol
    case {1} % TE
        
        for j = 1:length(lambda)
            A_air = exp(-im*kax(j)*d_air);
            A_mat = exp(-im*kmx(j)*d_mat);
            P_am = kmx(j)/kax(j);
            P_ma = kax(j)/kmx(j);
            B_am = 0.5*[(1+P_am).*A_mat, (1-P_am).*(A_mat)^-1; (1-P_am).*A_mat, (1+P_am).*(A_mat)^-1]; % Backward-propagation matrix
            B_ma = 0.5*[(1+P_ma).*A_air, (1-P_ma).*(A_air)^-1; (1-P_ma).*A_air, (1+P_ma).*(A_air)^-1];
            B_0 = 0.5*[(1+P_ma), (1-P_ma); (1-P_ma), (1+P_ma)];
            for ss = 1:length(layer)
                C = B_am*(B_ma*B_am)^(layer(ss))*B_0;
                t(j,ss) = 1/C(1,1); %transmission coefficient
                r(j,ss)=C(2,1)/C(1,1);
            end
        end
        
%     case {2} % TM
%         %% Transfer Matrix TM
%         
%         for j = 1:length(lambda)
%             A_air = exp(-im*k2x(j)*d_air);
%             A_mat = exp(-im*k1x(j)*d_mat);
%             P_am = k1x(j)/(n_mat.^2.*k0x(j));
%             P_ma = (n_mat.^2.*k2x(j))/k1x(j);
%             B_am = 0.5*[(1+P_am).*A_mat, (1-P_am).*(A_mat)^-1; (1-P_am).*A_mat, (1+P_am).*(A_mat)^-1]; % Backward-propagation matrix
%             B_ma = 0.5*[(1+P_ma).*A_air, (1-P_ma).*(A_air)^-1; (1-P_ma).*A_air, (1+P_ma).*(A_air)^-1];
%             B_0 = 0.5*[(1+P_ma), (1-P_ma); (1-P_ma), (1+P_ma)];
%             for ss = 1:length(layer)
%                 C = B_am*(B_ma*B_am)^(layer(ss))*B_0;
%                 t(j,ss) = 1/C(1,1); %transmission coefficient
%                 r(j,ss)=C(2,1)/C(1,1);
%             end
%         end
end


R = abs(r).^2;
T = abs(t).^2;
AA = R+T;
%
figure(1)
plot(c./f*1e9, R,'LineWidth',1.5);
% xlim([0.1 2])
xlabel('Frequency (THz)', FS, fs2)
ylabel('Reflectance (%)', FS, fs2)
title('Transfer Matrix', FS, fs3)
% h13 = legend('100nm','150nm','200nm','250nm','300nm');
% grid on;
set(gca,FS,16)

f_i = find(f==375e12); % 1 THz
% R_conv = R(f_i,:);
% 
% figure(2)
% plot(layer, R_conv,'LineWidth',1.5);
% % xlim([0.1 2])
% xlabel('N', FS, fs)
% ylabel('Reflectance (%)', FS, fs)
% % title('Transfer Matrix', FS, fs)
% % h13 = legend('100nm','150nm','200nm','250nm','300nm');
% % grid on;
% set(gca,FS,16)

M_r = r(f_i);

d = (1:1:1500)*1e-6;
g_th =(1./(1.*d).*log(1/(M_r^2)))/100; % [1/cm]

figure(3)
plot(d*1e6, abs(g_th),'LineWidth',1.5);
xlim([150 1500])
set(gca,'XTick',[150 300 450 600 750 900 1050 1200 1350 1500],'TickDir','in',FS,fs1,'fontweight','bold',LW,lw)
xlabel('Cavity length (um)', FS, fs2)
ylabel('Threshold gain, g_t_h (1/cm)', FS, fs2)
% title('Transfer Matrix', FS, fs)
% h13 = legend('100nm','150nm','200nm','250nm','300nm');
% grid on;
% set(gca,FS,16)

figure(5)
plot(c./freq*1e9, 1-abs(real(squeeze(L_t(k_i,n_i,:))))./1,LW,lw)
set(gca,'TickDir','in',FS,fs1,'fontweight','bold',LW,lw)
xlabel('Frequency (THz)',FS,fs2,'fontweight','bold')
ylabel('R',FS,fs2,'fontweight','bold')

% figure(4)
% plot(d*1e6, phase(g_th),'LineWidth',1.5);
% xlim([90 1000])
% xlabel('Cavity length (um)', FS, fs)
% ylabel('Threshold gain', FS, fs)
% % title('Transfer Matrix', FS, fs)
% % h13 = legend('100nm','150nm','200nm','250nm','300nm');
% % grid on;
% set(gca,FS,16)

U = M_r^2*exp(2*g_th.*d).*exp(im*2*kax(901).*d);
figure(5)
plot(d*1e6, abs(U),'LineWidth',1.5);
xlim([90 1000])
xlabel('Cavity length (um)', FS, fs2)
ylabel('Field intensity', FS, fs2)
% title('Transfer Matrix', FS, fs)
% h13 = legend('100nm','150nm','200nm','250nm','300nm');
% grid on;
set(gca,FS,16)

figure(6)
plot(d*1e6, mod(phase(U),2*pi),'LineWidth',1.5);
xlim([0 1000])
xlabel('Cavity length (um)', FS, fs2)
ylabel('Phase', FS, fs2)
% title('Transfer Matrix', FS, fs)
% h13 = legend('100nm','150nm','200nm','250nm','300nm');
% grid on;
set(gca,FS,16)


L_A_air = exp(-im*kax(f_i)*d_air);
L_A_mat = exp(-im*kmx(f_i)*d_mat);
L_A_cav = exp(-im*kcx(f_i)*d_gain);
L_P_am = kmx(f_i)/kax(f_i);
L_P_ma = kax(f_i)/kmx(f_i);
L_P_mc = kcx(f_i)/kmx(f_i);
L_P_cm = kmx(f_i)/kcx(f_i);
L_B_am = 0.5*[(1+L_P_am).*L_A_mat, (1-L_P_am).*(L_A_mat)^-1; (1-L_P_am).*L_A_mat, (1+L_P_am).*(L_A_mat)^-1]; % Backward-propagation matrix
L_B_ma = 0.5*[(1+L_P_ma).*L_A_air, (1-L_P_ma).*(L_A_air)^-1; (1-L_P_ma).*L_A_air, (1+L_P_ma).*(L_A_air)^-1];
L_B_mc = 0.5*[(1+L_P_mc).*L_A_cav, (1-L_P_mc).*(L_A_cav)^-1; (1-L_P_mc).*L_A_cav, (1+L_P_mc).*(L_A_cav)^-1];
L_B_cm = 0.5*[(1+L_P_cm).*L_A_mat, (1-L_P_cm).*(L_A_mat)^-1; (1-L_P_cm).*L_A_mat, (1+L_P_cm).*(L_A_mat)^-1];
L_B_0 = 0.5*[(1+L_P_ma), (1-L_P_ma); (1-L_P_ma), (1+L_P_ma)];
for ss = 1:length(layer)
    L_C = L_B_am*(L_B_ma*L_B_am)^(layer(ss))*L_B_mc*L_B_cm*(L_B_ma*L_B_am)^(layer(ss))*L_B_0;
    L_t(ss) = 1/L_C(1,1); %transmission coefficient
    L_r(ss)= L_C(2,1)/L_C(1,1);
end
% cav_lay = 
% 
% total_layer = (layer*2+1)*2+3;
% E_end = L_t*E_in; %% 33번째 마지막 layer - air
% for l = 1:total_layer
%     E_temp = E_end*