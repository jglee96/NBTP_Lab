close all;

x = 0:0.001:10;
k_comp = 100000;
d = 737.5*1e-6;

figure(1);
plot(x,real((4.*x).*exp(-x.*1i*d*20)./(x+1).^2),x,imag((4.*x).*exp(-x.*1i*d*20)./(x+1).^2),'linewidth',2);
hold on;
plot(x,real((4.*x).*exp(-x.*1i*d*k_comp)./(x+1).^2),x,imag((4.*x).*exp(-x.*1i*d*k_comp)./(x+1).^2),'linewidth',2);
grid on;
ylim([-3 1.5]);
legend('real(n),k=20', 'imag(n),k=20',['real(n),k=',num2str(k_comp)],['imag(n),k=',num2str(k_comp)]);
set(gca,'Fontsize',30);
set(gca,'XTick',0:1:10)
xlabel('Index');
ylabel('Transmission');

figure(2);
plot(f,abs(T1_th),'linewidth',2);
xlim([0.01 2]);
%ylim([-0.8 0.8]);
%legend('real(T_{ex})', 'imag(T_{ex})');
set(gca,'Fontsize',30);
xlabel('Frequency [THz]');
ylabel('Transmission');

% t12 = (2*1)./(x+1);
% t23 = (2.*x)./(x+1);
% r23 = (x-1)./(x+1);
% 
% T1 = t12.*t23.*exp(-1i.*x*d*100000);
% T2 = t12.*(r23.^2).*(t23).*exp(-3i.*x*d*100000);
% T3 = t12.*(r23.^4).*(t23).*exp(-5i.*x*d*100000);
% T = T1+T2+T3;
% 
% figure(3)
% plot(x,abs(T));
