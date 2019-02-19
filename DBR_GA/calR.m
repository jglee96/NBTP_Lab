function R = calR(DBR_eps,DBR_length,lambda,L)
% DBR Calcultor

epst = DBR_eps;
lengtht = DBR_length;
B_tot = zeros(2,2,length(lambda));

% calculate transfer matrix (B_tot)
for nl = 1:L+1
    for l=1:length(lambda)
        
        Bf = @(P,kx,h) (1/2)*[(1+P)*exp(-1i*kx*h) (1-P)*exp(1i*kx*h); ...
            (1-P)*exp(-1i*kx*h) (1+P)*exp(1i*kx*h)];
        
        if nl == 1
            P = sqrt(epst(nl)/1);
            h = lengtht(nl);
            kx = sqrt(epst(nl))*2*pi/(lambda(l));
            B_tot(:,:,l) = Bf(P,kx,h);
        elseif nl == L+1
            P = sqrt(1/epst(nl-1));
            h = 0;
            kx = 1*2*pi/(lambda(l));
            B_tot(:,:,l) = squeeze(B_tot(:,:,l))*Bf(P,kx,h);
        else
            P = sqrt(epst(nl)/epst(nl-1));
            h = lengtht(nl);
            kx = sqrt(epst(nl))*2*pi/(lambda(l));
            B_tot(:,:,l) = squeeze(B_tot(:,:,l))*Bf(P,kx,h);
        end
    end
end
R(1,:) = abs((B_tot(2,1,:)./B_tot(1,1,:))).^2;
