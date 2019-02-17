%%% 1D Distributed Bragg Reflctor with Genetic Algorithm %%%
clc; clear; close all;

N = 7; % Number of permitivitty layers
N_struct = 3; % Number of DBR structures
L = 2*N-1; % Number of total layers
tens = 1000; %
epsi = 12.25; % layer permitivitty
eps0 = 1; % air permitivitty

% Setup target lambda region
minlam = 500;
maxlam = 2000;
lambda = linspace(minlam, maxlam, (maxlam-minlam)+1);


DBR_eps = ones(N_struct,L); % DBR layers permitivitty
DBR_length = ones(N_struct,L); % DBR layers length

thick = 200*1e-6;
for i=1:N_struct
    for j=1:L
        % Set DBR layers length
        a = rand*tens;
        while a == 0
            a = rand*tens;
        end
        DBR_length(i,j) = a*1e-6;
        % Set DBR layers permitivitty
        if mod(j,2)
            DBR_eps(i,j) = epsi;
            DBR_length(N_struct,j) = thick/epsi;
        else
            DBR_eps(i,j) = eps0;
            DBR_length(N_struct,j) = thick/eps0;
        end
    end
end


% DBR Calcultor
R = zeros(N_struct,length(lambda));
T = zeros(N_struct,length(lambda));

for i=1:N_struct    
    epst = DBR_eps(i,:);
    lengtht = DBR_length(i,:);
    B_tot = zeros(2,2,length(lambda));
    
    % calculate transfer matrix (B_tot)
    for nl = 1:L+1
        for l=1:length(lambda)
            
            Bf = @(P,kx,h) (1/2)*[(1+P)*exp(-1i*kx*h) (1-P)*exp(1i*kx*h); ...
                (1-P)*exp(-1i*kx*h) (1+P)*exp(1i*kx*h)];            
            
            if nl == 1                
                P = sqrt(epst(nl)/1);
                h = lengtht(nl);
                kx = sqrt(epst(nl))*2*pi/(lambda(l)*1e-6);
                B_tot(:,:,l) = Bf(P,kx,h);
            elseif nl == L+1
                P = sqrt(1/epst(nl-1));
                h = 0;
                kx = 1*2*pi/(lambda(l)*1e-6);
                B_tot(:,:,l) = squeeze(B_tot(:,:,l))*Bf(P,kx,h);                
            else
                P = sqrt(epst(nl)/epst(nl-1));
                h = lengtht(nl)-lengtht(nl-1);
                kx = sqrt(epst(nl))*2*pi/(lambda(l)*1e-6);
                B_tot(:,:,l) = squeeze(B_tot(:,:,l))*Bf(P,kx,h);             
            end              
        end
    end
    T(i,:) = abs((1./B_tot(1,1,:))).^2;
    R(i,:) = abs((B_tot(2,1,:)./B_tot(1,1,:))).^2;
end

figure(1)
for i=1:N_struct
    subplot(N_struct,1,i);
    plot(lambda, T(i,:));
end

figure(2)
for i=1:N_struct
    subplot(N_struct,1,i);
    plot(lambda, R(i,:));
end