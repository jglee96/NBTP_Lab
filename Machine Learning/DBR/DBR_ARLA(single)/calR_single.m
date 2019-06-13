function R = calR_single(Layer,lambda,Ngrid,dx,epsi,eps0)
% DBR Calcultor
one_element = epsi*double((Layer == 1));
zero_element = eps0*double((Layer == 0));
epst = single(one_element + zero_element);

% calculate transfer matrix (B_tot)
Pn = [epst, 1];
Pn1 = [1, epst];
P = arrayfun(@(pn,pn1) sqrt(pn./pn1), Pn, Pn1);
kx = zeros(length(lambda),Ngrid+1);
for l=1:length(lambda)
    kx(l,:) = sqrt(Pn)*2*pi/lambda(l);
end
h = [dx*ones(1,Ngrid), 0];
B11 = zeros(Ngrid+1,length(lambda),'single');
B12 = zeros(Ngrid+1,length(lambda),'single');
B21 = zeros(Ngrid+1,length(lambda),'single');
B22 = zeros(Ngrid+1,length(lambda),'single');

for l=1:length(lambda)
    B11(:,l) = arrayfun(@(p,kx,h) (1+p).*exp(-1i.*kx.*h),P,kx(l,:),h);
    B12(:,l) = arrayfun(@(p,kx,h) (1-p).*exp(1i.*kx.*h),P,kx(l,:),h);
    B21(:,l) = arrayfun(@(p,kx,h) (1-p).*exp(-1i.*kx.*h),P,kx(l,:),h);
    B22(:,l) = arrayfun(@(p,kx,h) (1+p).*exp(1i.*kx.*h),P,kx(l,:),h);
end

B = cell(Ngrid+1,length(lambda));
for l=1:length(lambda)
    B(:,l) = arrayfun(@(b11,b12,b21,b22) (1/2)*[b11 b12;b21 b22],B11(:,l),B12(:,l),B21(:,l),B22(:,l),'UniformOutput',false);
end

R = zeros(1,length(lambda),'single');
for l=1:length(lambda)
    Btot = eye(2);
    for j=1:Ngrid+1
        Btot = Btot*cell2mat(B(j,l));
    end
    R(l) = abs(Btot(2,1)/Btot(1,1))^2;
end
