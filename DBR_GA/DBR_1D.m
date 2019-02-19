%%% 1D Distributed Bragg Reflctor with Genetic Algorithm %%%
clc; clear; close all;

dx = 1; % DBR minimum length element in nm
L = 794*dx; % DBR Length in nm
eps0 = 12.25; % DBR material permitivitty
np = 2; % number of intial parents

% Setup target lambda region
minlam = 100*dx;
maxlam = 1000*dx;
lambda = linspace(minlam, maxlam, (maxlam-minlam)/dx+1);

% Setup DBR
DBR = ones(np,L);
N = int16(L*0.3);

for i=1:np
    for j=1:N
        a = int16(rand*L);
        while a <= 0
            a = int16(rand*L);
        end
        DBR(i,a) = eps0;
    end
end

thick = 100;
p = 1;
for i=1:L
   DBR(np,i) = eps0;
   p = p+1;
   if p > int16(thick/sqrt(eps0))
       DBR(np,i) = 1;
       p = p+1;
       if p > int16(thick/sqrt(eps0))+thick
           p = 1;
       end
   end   
end

DBR(:,1) = eps0;
DBR(:,L) = eps0;

% DBR Calcultor
R = zeros(np,length(lambda));
T = zeros(np,length(lambda));

for i=1:np
    
    epsi = DBR(i,:);
    B_tot = ones(2,2,length(lambda));
    
    % find initial condition
    d = 1;
    while epsi(d+1) == epsi(d)
        d = d+1;
    end
    
    for l=1:length(lambda)
        d0 = d;
        Bf = @(P,kx,h) (1/2)*[(1+P)*exp(-1i*kx*h) (1-P)*exp(1i*kx*h); ...
            (1-P)*exp(-1i*kx*h) (1+P)*exp(1i*kx*h)];
        % calculate transfer matrix (B_tot)
        P = sqrt(epsi(1)/1);
        kx = sqrt(epsi(1))*2*pi/(lambda(l));
        h=d0;
        B = Bf(P,kx,h);
        B_tot(:,:,l) = B;
        
        j = 1;
        while j < L+1
            d1 = 1;
            while (j+d1+1 < L) && (epsi(j+d1+1) == epsi(j+d1))
                d1 = d1+1;
            end
            
            if j+1 > L % the last L+1 region
                P = sqrt(1/epsi(L));
                B = (1/2)*[(1+P) (1-P); ...
                    (1-P) (1+P)];
                B_tot(:,:,l) = squeeze(B_tot(:,:,l))*B;
            else % the DBR region
                P = sqrt(epsi(j+1)/epsi(j));
                h = d1;
                kx = sqrt(epsi(j+1))*2*pi/(lambda(l));
                B = Bf(P,kx,h);
                B_tot(:,:,l) = squeeze(B_tot(:,:,l))*B;
            end
            
            j = j+d1;
        end
    end
    T(i,:) = abs((1./B_tot(1,1,:))).^2;
    R(i,:) = abs((B_tot(2,1,:)./B_tot(1,1,:))).^2;
end

figure(1)
for i=1:np
    subplot(np,1,i);
    plot(lambda, T(i,:));
end

figure(2)
for i=1:np
    subplot(np,1,i);
    plot(lambda, R(i,:));
end

figure(3)
for i=1:np
    subplot(np,1,i);
    bar(dx:dx:L*dx, DBR(i,:),1.0);
end
