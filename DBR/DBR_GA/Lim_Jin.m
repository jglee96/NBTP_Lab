clear all
clc
close all

FS = 'FontSize'; fs1 = 16; fs2 = 18; fs3 = 20;
LS = 'linestyle'; NN ='none';
LN = 50;
LW = 'linewidth'; lw=1.5;
LV = 10;
c=3e8;
freq = (0:0.0000001:2.0)*1e12;
lambda = 300e-6;

%% cavity material
% n = 1:0.1:3;
n=1;
% k = -0.1:0.05:0.1;
k=0;
[A,B]=meshgrid(n,k);
d_cav = 1*lambda/2./n; % cavity length
%% Air
n_air = 1; % air index
k_air = 2*pi./lambda; % air wave vector
d_air = n_air*lambda/4;
%% Mirror material
n_mat = 1.95;%+1i*0.00525; % material index
k_mat = n_mat*2*pi./lambda; % material wave vector
d_mat = lambda/4/real(n_mat);
layer = 4; % Mirror material 偎熱

for f = 1:length(freq)
    for l = 1:length(n)
        for j = 1:length(k)
            k_cav(j,l,f) = 2*pi*freq(f)./c*(n(l)+1i*k(j));
        end
    end
end
alpha_in = 2*pi/lambda*B;
% alpha_in = 0.22;

kax = k_air; kmx = k_mat; kcx = k_cav;
for f = 1:length(freq)
    for l = 1:length(n)
        for j = 1:length(k)
            A_air = exp(-1i*kax*d_air);
            A_mat = exp(-1i*kmx*d_mat);
            A_cav = exp(-1i*kcx(j,l,f)*d_cav(l));
            P_cm = kmx./kcx(j,l,f);
            P_mc = kcx(j,l,f)./kmx;
            P_am = kmx/kax;
            P_ma = kax/kmx;
            B_mc = 0.5*[(1+P_mc).*A_cav, (1-P_mc).*(A_cav)^-1; (1-P_mc).*A_cav, (1+P_mc).*(A_cav)^-1];
            B_cm = 0.5*[(1+P_cm).*A_mat, (1-P_cm).*(A_mat)^-1; (1-P_cm).*A_mat, (1+P_cm).*(A_mat)^-1]; % Backward-propagation matrix
            B_am = 0.5*[(1+P_am).*A_mat, (1-P_am).*(A_mat)^-1; (1-P_am).*A_mat, (1+P_am).*(A_mat)^-1]; % Backward-propagation matrix
            B_ma = 0.5*[(1+P_ma).*A_air, (1-P_ma).*(A_air)^-1; (1-P_ma).*A_air, (1+P_ma).*(A_air)^-1];
            B_0 = 0.5*[(1+P_ma), (1-P_ma); (1-P_ma), (1+P_ma)];
            
            C = B_cm*(B_ma*B_am)^(3)*B_0; %() layer 熱
            L_C = B_am*(B_ma*B_am)^(layer)*B_mc*B_cm*(B_ma*B_am)^(layer)*B_0;
            t(j,l,f) = 1/C(1,1); %transmission coefficient
            r(j,l,f)=C(2,1)/C(1,1); %reflection coefficient
            L_t(j,l,f) = 1/L_C(1,1);
            L_r(j,l,f)= L_C(2,1)/L_C(1,1);
            
        end
    end
end
for f=1:length(freq)
    g_th(:,:,f) = (alpha_in + 1./(1.*d_cav).*log(1./(abs(r(:,:,f)).^2)))./100; % [1/cm]
end

f_i = find(freq==1e12);
k_i = find(k==k);
n_i = find(n==n);
figure(1)
imagesc(n,k,g_th(:,:,f_i))
set(gca,'ydir','normal','xdir','normal','TickDir','in',FS,fs1,'fontweight','bold',LW,lw)
% set(gca,'ydir','normal','xdir','normal','XTick',[0 50 100],'XTicklabel',[],'YTick',[0 50 100 150 200],'TickDir','in',FS,fs1,'fontweight','bold',LW,lw)
% axis image
% xlim([12,15])
% ylim([0,8])
% caxis([42, 65])
title('Threshold gain, g_t_h [1/cm]',FS, fs3,'fontweight','bold')
xlabel('Refractive index, n',FS,fs2,'fontweight','bold')
ylabel('Extinction coefficient, k',FS,fs2,'fontweight','bold')
colormap(jet(256))
colorbar

figure(2)
plot(n,g_th(k_i,:,f_i),LW,lw)
set(gca,'TickDir','in',FS,fs1,'fontweight','bold',LW,lw)
xlabel('Refractive index, n',FS,fs2,'fontweight','bold')
ylabel('Threshold gain, g_t_h (1/cm)',FS,fs2,'fontweight','bold')
title('Extinction coefficient, k = 0',FS,fs2,'fontweight','bold')

figure(3)
plot(freq*1e-12,abs(real(squeeze(L_t(k_i,n_i,:))))./1,LW,lw)
set(gca,'TickDir','in',FS,fs1,'fontweight','bold',LW,lw)
xlabel('Frequency (THz)',FS,fs2,'fontweight','bold')
ylabel('Transmission',FS,fs2,'fontweight','bold')
% title('Extinction coefficient, k = 0',FS,fs2,'fontweight','bold')

% Res = (abs(squeeze(L_t(k_i,n_i,f_i)))).^1;
% temp1 = find((abs(squeeze(L_t(k_i,n_i,1:f_i)))).^1<=Res/2);
% FWHM_L = max(temp1);
% temp2 = find((abs(squeeze(L_t(k_i,n_i,f_i:2*f_i)))).^1+f_i<=Res/2);
% FWHM_R = min(temp2);
% Q = freq(f_i)./(freq(FWHM_R)-freq(FWHM_L));

g_th = squeeze(g_th(:,:,f_i));
AA = squeeze(L_t);
BB = squeeze(abs(r));

% clear all
% clc
% close all
% 
% FS = 'FontSize'; fs1 = 16; fs2 = 18; fs3 = 20;
% LS = 'linestyle'; NN ='none';
% LN = 50;
% LW = 'linewidth'; lw=1.5;
% LV = 10;
% c=3e8;
% freq = (150:1:1000)*1e12;
% lambda = 800e-9;
% 
% %% cavity material
% % n = 1:0.1:3;
% n=1;
% % k = -0.1:0.05:0.1;
% k=0;
% [A,B]=meshgrid(n,k);
% d_cav = 1*lambda/2./n; % cavity length
% %% Air
% n_air = 1; % air index
% k_air = 2*pi./lambda; % air wave vector
% d_air = n_air*lambda/4;
% %% Mirror material
% n_mat = 1.95;%+1i*0.00525; % material index
% k_mat = n_mat*2*pi./lambda; % material wave vector
% d_mat = lambda/4/real(n_mat);
% layer = 4; % Mirror material 偎熱
% 
% for f = 1:length(freq)
%     for l = 1:length(n)
%         for j = 1:length(k)
%             k_cav(j,l,f) = 2*pi*freq(f)./c*(n(l)+1i*k(j));
%         end
%     end
% end
% alpha_in = 2*pi/lambda*B;
% % alpha_in = 0.22;
% 
% kax = k_air; kmx = k_mat; kcx = k_cav;
% for f = 1:length(freq)
%     for l = 1:length(n)
%         for j = 1:length(k)
%             A_air = exp(-1i*kax*d_air);
%             A_mat = exp(-1i*kmx*d_mat);
%             A_cav = exp(-1i*kcx(j,l,f)*d_cav(l));
%             P_cm = kmx./kcx(j,l,f);
%             P_mc = kcx(j,l,f)./kmx;
%             P_am = kmx/kax;
%             P_ma = kax/kmx;
%             B_mc = 0.5*[(1+P_mc).*A_cav, (1-P_mc).*(A_cav)^-1; (1-P_mc).*A_cav, (1+P_mc).*(A_cav)^-1];
%             B_cm = 0.5*[(1+P_cm).*A_mat, (1-P_cm).*(A_mat)^-1; (1-P_cm).*A_mat, (1+P_cm).*(A_mat)^-1]; % Backward-propagation matrix
%             B_am = 0.5*[(1+P_am).*A_mat, (1-P_am).*(A_mat)^-1; (1-P_am).*A_mat, (1+P_am).*(A_mat)^-1]; % Backward-propagation matrix
%             B_ma = 0.5*[(1+P_ma).*A_air, (1-P_ma).*(A_air)^-1; (1-P_ma).*A_air, (1+P_ma).*(A_air)^-1];
%             B_0 = 0.5*[(1+P_ma), (1-P_ma); (1-P_ma), (1+P_ma)];
%             
%             C = B_cm*(B_ma*B_am)^(3)*B_0; %() layer 熱
%             L_C = B_am*(B_ma*B_am)^(layer)*B_mc*B_cm*(B_ma*B_am)^(layer)*B_0;
%             t(j,l,f) = 1/C(1,1); %transmission coefficient
%             r(j,l,f)=C(2,1)/C(1,1); %reflection coefficient
%             L_t(j,l,f) = 1/L_C(1,1);
%             L_r(j,l,f)= L_C(2,1)/L_C(1,1);
%             
%         end
%     end
% end
% for f=1:length(freq)
%     g_th(:,:,f) = (alpha_in + 1./(1.*d_cav).*log(1./(abs(r(:,:,f)).^2)))./100; % [1/cm]
% end
% 
% f_i = find(freq==375e12);
% k_i = find(k==k);
% n_i = find(n==n);
% figure(1)
% imagesc(n,k,g_th(:,:,f_i))
% set(gca,'ydir','normal','xdir','normal','TickDir','in',FS,fs1,'fontweight','bold',LW,lw)
% % set(gca,'ydir','normal','xdir','normal','XTick',[0 50 100],'XTicklabel',[],'YTick',[0 50 100 150 200],'TickDir','in',FS,fs1,'fontweight','bold',LW,lw)
% % axis image
% % xlim([12,15])
% % ylim([0,8])
% % caxis([42, 65])
% title('Threshold gain, g_t_h [1/cm]',FS, fs3,'fontweight','bold')
% xlabel('Refractive index, n',FS,fs2,'fontweight','bold')
% ylabel('Extinction coefficient, k',FS,fs2,'fontweight','bold')
% colormap(jet(256))
% colorbar
% 
% figure(2)
% plot(n,g_th(k_i,:,f_i),LW,lw)
% set(gca,'TickDir','in',FS,fs1,'fontweight','bold',LW,lw)
% xlabel('Refractive index, n',FS,fs2,'fontweight','bold')
% ylabel('Threshold gain, g_t_h (1/cm)',FS,fs2,'fontweight','bold')
% title('Extinction coefficient, k = 0',FS,fs2,'fontweight','bold')
% 
% figure(3)
% plot(freq*1e-12,abs(real(squeeze(L_t(k_i,n_i,:))))./1,LW,lw)
% set(gca,'TickDir','in',FS,fs1,'fontweight','bold',LW,lw)
% xlabel('Frequency (THz)',FS,fs2,'fontweight','bold')
% ylabel('Transmission',FS,fs2,'fontweight','bold')
% % title('Extinction coefficient, k = 0',FS,fs2,'fontweight','bold')
% 
% 
% figure(5)
% plot(c./freq*1e9, 1-abs(real(squeeze(L_t(k_i,n_i,:))))./1,LW,lw)
% set(gca,'TickDir','in',FS,fs1,'fontweight','bold',LW,lw)
% xlabel('Frequency (THz)',FS,fs2,'fontweight','bold')
% ylabel('R',FS,fs2,'fontweight','bold')
% 
% 
% % Res = (abs(squeeze(L_t(k_i,n_i,f_i)))).^1;
% % temp1 = find((abs(squeeze(L_t(k_i,n_i,1:f_i)))).^1<=Res/2);
% % FWHM_L = max(temp1);
% % temp2 = find((abs(squeeze(L_t(k_i,n_i,f_i:2*f_i)))).^1+f_i<=Res/2);
% % FWHM_R = min(temp2);
% % Q = freq(f_i)./(freq(FWHM_R)-freq(FWHM_L));
% 
% g_th = squeeze(g_th(:,:,f_i));
% AA = squeeze(L_t);
% BB = squeeze(abs(r));