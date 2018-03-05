close all
clear all
clc

c0=299792458;
d=737.5.*1e-6;
L1=0;

A=zeros(3000,2);
A_temp=load('measured_signal.txt');
A(1:1250,1)=A_temp(1:1250,2);
A(1:1250,2)=A_temp(1:1250,3);
t_span=A_temp(:,1).*1e-12;

% A(1200:length(t),2)=0;
% A(1200:length(t),3)=0;

ref=A(:,1);
sam=A(:,2);

dt=t_span(2)-t_span(1);
t= 0:dt:(length(A(:,1))-1)*dt;

f_span=1./dt;

L=length(A(:,2));
f=linspace(0,f_span,L+1);
f(L+1)=[];

R=fft(ref);
S=fft(sam);
R_amp=abs(R);
R_phase=phase(R);
S_amp=abs(S);
S_phase=phase(S);

n_ini=3.4+1i;
n(1:length(f))=n_ini;
Z(1:length(f))=1;

for l=1:length(f)
    
    k0(l)=2.*pi.*f(l)./c0;
    T(l)=S(l)./(R(l).*exp(1i.*k0(l).*d));
%     T=transpose(T);
    
end

for l=300:length(f)
    
    n(l)=n(l-1);

     
    while abs(Z(l))>1e-10
        
       
%         r2(l)=(n(l)-1)./(n(l)+1);
        t1(l)=2./(n(l)+1);
        t2(l)=2.*n(l)./(n(l)+1);
        
        q(l)=t1(l).*t2(l).*exp(-1i.*k0(l).*n(l).*d);%./(1-((r2(l)).^2).*exp(-1i.*2.*k0(l).*n(l).*d));%.*(1-(((r2(l)).^2).*exp(-1i.*2.*k0(l).*n(l).*d)).^1);
        
        Z(l)=q(l)-T(l);
        Y(l)=(4*exp(-(pi*d*f(l)*n(l)*1i)/149896229))/(n(l) + 1)^2 - (8*n(l)*exp(-(pi*d*f(l)*n(l)*1i)/149896229))/(n(l) + 1)^3 - (pi*d*f(l)*n(l)*exp(-(pi*d*f(l)*n(l)*1i)/149896229)*4i)/(149896229*(n(l) + 1)^2);
%         Y(l)=(4*exp(-(pi*f(l)*n(l)*59*i)/11991698320000)*((exp(-(pi*f(l)*n(l)*177*i)/5995849160000)*(n(l) - 1)^6)/(n(l) + 1)^6 - 1))/(((exp(-(pi*f(l)*n(l)*59*i)/5995849160000)*(n(l) - 1)^2)/(n(l) + 1)^2 - 1)*(n(l) + 1)^2) - (8*n(l)*exp(-(pi*f(l)*n(l)*59*i)/11991698320000)*((exp(-(pi*f(l)*n(l)*177*i)/5995849160000)*(n(l) - 1)^6)/(n(l) + 1)^6 - 1))/(((exp(-(pi*f(l)*n(l)*59*i)/5995849160000)*(n(l) - 1)^2)/(n(l) + 1)^2 - 1)*(n(l) + 1)^3) - (4*n(l)*exp(-(pi*f(l)*n(l)*59*i)/11991698320000)*(- (6*exp(-(pi*f(l)*n(l)*177*i)/5995849160000)*(n(l) - 1)^5)/(n(l) + 1)^6 + (6*exp(-(pi*f(l)*n(l)*177*i)/5995849160000)*(n(l) - 1)^6)/(n(l) + 1)^7 + (pi*f(l)*exp(-(pi*f(l)*n(l)*177*i)/5995849160000)*(n(l) - 1)^6*177*i)/(5995849160000*(n(l) + 1)^6)))/(((exp(-(pi*f(l)*n(l)*59*i)/5995849160000)*(n(l) - 1)^2)/(n(l) + 1)^2 - 1)*(n(l) + 1)^2) + (4*n(l)*exp(-(pi*f(l)*n(l)*59*i)/11991698320000)*((exp(-(pi*f(l)*n(l)*177*i)/5995849160000)*(n(l) - 1)^6)/(n(l) + 1)^6 - 1)*(- (exp(-(pi*f(l)*n(l)*59*i)/5995849160000)*(2*n(l) - 2))/(n(l) + 1)^2 + (2*exp(-(pi*f(l)*n(l)*59*i)/5995849160000)*(n(l) - 1)^2)/(n(l) + 1)^3 + (pi*f(l)*exp(-(pi*f(l)*n(l)*59*i)/5995849160000)*(n(l) - 1)^2*59*i)/(5995849160000*(n(l) + 1)^2)))/(((exp(pi*f(l)*n(l)*(-(59*i)/5995849160000))*(n(l) - 1)^2)/(n(l) + 1)^2 - 1)^2*(n(l) + 1)^2) - (pi*f(l)*n(l)*exp(-(pi*f(l)*n(l)*59*i)/11991698320000)*((exp(-(pi*f(l)*n(l)*177*i)/5995849160000)*(n(l) - 1)^6)/(n(l) + 1)^6 - 1)*59*i)/(2997924580000*((exp(-(pi*f(l)*n(l)*59*i)/5995849160000)*(n(l) - 1)^2)/(n(l) + 1)^2 - 1)*(n(l) + 1)^2);
       
        n(l)=n(l)-Z(l)./Y(l);
   
    end
end
% 
for l=299:-1:3
    
    n(l)=n(l+1);
     
    while abs(Z(l))>1e-10
        
       
%         r2(l)=(n(l)-1)./(n(l)+1);
        t1(l)=2./(n(l)+1);
        t2(l)=2.*n(l)./(n(l)+1);
        
        q(l)=t1(l).*t2(l).*exp(-1i.*k0(l).*n(l).*d);%./(1-((r2(l)).^2).*exp(-1i.*2.*k0(l).*n(l).*d));%.*(1-(((r2(l)).^2).*exp(-1i.*2.*k0(l).*n(l).*d)).^1);
        
        Z(l)=q(l)-T(l);
        Y(l)=(4*exp(-(pi*d*f(l)*n(l)*1i)/149896229))/(n(l) + 1)^2 - (8*n(l)*exp(-(pi*d*f(l)*n(l)*1i)/149896229))/(n(l) + 1)^3 - (pi*d*f(l)*n(l)*exp(-(pi*d*f(l)*n(l)*1i)/149896229)*4i)/(149896229*(n(l) + 1)^2);
%         Y(l)=(4*exp(-(pi*f(l)*n(l)*59*i)/11991698320000)*((exp(-(pi*f(l)*n(l)*177*i)/5995849160000)*(n(l) - 1)^6)/(n(l) + 1)^6 - 1))/(((exp(-(pi*f(l)*n(l)*59*i)/5995849160000)*(n(l) - 1)^2)/(n(l) + 1)^2 - 1)*(n(l) + 1)^2) - (8*n(l)*exp(-(pi*f(l)*n(l)*59*i)/11991698320000)*((exp(-(pi*f(l)*n(l)*177*i)/5995849160000)*(n(l) - 1)^6)/(n(l) + 1)^6 - 1))/(((exp(-(pi*f(l)*n(l)*59*i)/5995849160000)*(n(l) - 1)^2)/(n(l) + 1)^2 - 1)*(n(l) + 1)^3) - (4*n(l)*exp(-(pi*f(l)*n(l)*59*i)/11991698320000)*(- (6*exp(-(pi*f(l)*n(l)*177*i)/5995849160000)*(n(l) - 1)^5)/(n(l) + 1)^6 + (6*exp(-(pi*f(l)*n(l)*177*i)/5995849160000)*(n(l) - 1)^6)/(n(l) + 1)^7 + (pi*f(l)*exp(-(pi*f(l)*n(l)*177*i)/5995849160000)*(n(l) - 1)^6*177*i)/(5995849160000*(n(l) + 1)^6)))/(((exp(-(pi*f(l)*n(l)*59*i)/5995849160000)*(n(l) - 1)^2)/(n(l) + 1)^2 - 1)*(n(l) + 1)^2) + (4*n(l)*exp(-(pi*f(l)*n(l)*59*i)/11991698320000)*((exp(-(pi*f(l)*n(l)*177*i)/5995849160000)*(n(l) - 1)^6)/(n(l) + 1)^6 - 1)*(- (exp(-(pi*f(l)*n(l)*59*i)/5995849160000)*(2*n(l) - 2))/(n(l) + 1)^2 + (2*exp(-(pi*f(l)*n(l)*59*i)/5995849160000)*(n(l) - 1)^2)/(n(l) + 1)^3 + (pi*f(l)*exp(-(pi*f(l)*n(l)*59*i)/5995849160000)*(n(l) - 1)^2*59*i)/(5995849160000*(n(l) + 1)^2)))/(((exp(pi*f(l)*n(l)*(-(59*i)/5995849160000))*(n(l) - 1)^2)/(n(l) + 1)^2 - 1)^2*(n(l) + 1)^2) - (pi*f(l)*n(l)*exp(-(pi*f(l)*n(l)*59*i)/11991698320000)*((exp(-(pi*f(l)*n(l)*177*i)/5995849160000)*(n(l) - 1)^6)/(n(l) + 1)^6 - 1)*59*i)/(2997924580000*((exp(-(pi*f(l)*n(l)*59*i)/5995849160000)*(n(l) - 1)^2)/(n(l) + 1)^2 - 1)*(n(l) + 1)^2);
       
        n(l)=n(l)-Z(l)./Y(l);
   
    end
end


FS = 'Fontsize';
LW = 'LineWidth';
fs1 = 14;
fs2 = 16;
lw = 1;

figure(1)
h = plot(t*1e12,ref,t*1e12,sam);%'-b',LW,lw
% set(h(1),'Color',[0 0 1])
% set(h(2),'Color',[1 0 0])
% h(1).LineStyle = '--';
% set(gca,FS,fs1,LW,1.7,'XTick',[0.6 1 1.4 1.8 2.20],'YTick',[-0.05 -0.03 -0.01 0.01],'TickDir','in','Fontname','Arial')
% xlabel('Frequency(THz)',FS,fs2)
% ylabel('Extinction Coefficient',FS,fs2)
xlim([0 80])
% ylim([3.39 3.415])
% legend('n ini(w)=2')
% set(legend,FS,fs1)

figure(2)
h = plot(f*1e-12,abs(T));%'-b',LW,lw
% set(h(1),'Color',[0 0 1])
% set(h(2),'Color',[1 0 0])
% h(1).LineStyle = '--';
% set(gca,FS,fs1,LW,1.7,'XTick',[0.6 1 1.4 1.8 2.20],'YTick',[-0.05 -0.03 -0.01 0.01],'TickDir','in','Fontname','Arial')
% xlabel('Frequency(THz)',FS,fs2)
% ylabel('Extinction Coefficient',FS,fs2)
xlim([0.2 2.2])
% ylim([3.39 3.415])
% legend('n ini(w)=2')
% set(legend,FS,fs1)

figure(3)
h = plot(f*1e-12,real(n),f*1e-12,-imag(n));%'-b',LW,lw,
% set(h(1),'Color',[0 0 1])
% set(h(2),'Color',[1 0 0])
% h(1).LineStyle = '--';
% set(gca,FS,fs1,LW,1.7,'XTick',[0.6 1 1.4 1.8 2.20],'YTick',[-0.05 -0.03 -0.01 0.01],'TickDir','in','Fontname','Arial')
% xlabel('Frequency(THz)',FS,fs2)
% ylabel('Extinction Coefficient',FS,fs2)
xlim([0 2.5])
% ylim([3.39 3.415])
% legend('n ini(w)=2')
% set(legend,FS,fs1)

