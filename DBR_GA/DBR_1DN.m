%%% 1D Distributed Bragg Reflctor with Genetic Algorithm %%%
clc; clear; close all;

% Setup program constants
N = 6; % Number of permitivitty layers
N_struct = 3; % Number of DBR structures
L = 2*N-1; % Number of total layers
tens = 1000; %
epsi = 12.25; % layer permitivitty
eps0 = 1; % air permitivitty
tarlam = 800; % target wavelength

% Setup target lambda region
minlam = 400; % minimum wavelength
maxlam = 1200; % maximum wavelength
lambda = linspace(minlam, maxlam, (maxlam-minlam)+1); % wavelength vector


DBR_eps = ones(1,L); % DBR layers permitivitty
DBR_length = ones(N_struct,L); % DBR layers length

thick = 200;
for i=1:N_struct
    for j=1:L
        % Set DBR layers length
        a = rand*tens;
        while a == 0
            a = rand*tens;
        end
        DBR_length(i,j) = a;
        % Set DBR layers permitivitty
        if mod(j,2)
            DBR_eps(j) = epsi;
        else
            DBR_eps(j) = eps0;
        end
        %         DBR_length(N_struct,j) = thick/sqrt(DBR_eps(j)); % theory result
    end
end

% Initial DBR Calcultor
R = zeros(N_struct,length(lambda));
Q = zeros(1,N_struct);
for i=1:N_struct
    R(i,:)= calR(DBR_eps,DBR_length(i,:),lambda,L);
   % Q(i) = calQ(R(i,:),lambda,tarlam);
end
delindex = find(~Q);
DBR_length(delindex,:) = [];
R(delindex,:) = [];
%Q(delindex) = [];
Plot_R(R,lambda);

tarlam_index = find(lambda==tarlam);

% Crossover
Ng = 5;
for n=1:Ng
    
    N_struct = length(DBR_length(:,1));
    R = zeros(N_struct,length(lambda));
    Q = zeros(1,N_struct);
    for i=1:N_struct
        R(i,:)= calR(DBR_eps,DBR_length(i,:),lambda,L);
        Q(i) = calQ(R(i,:),lambda,tarlam);
    end
    delindex = find(~Q);
    DBR_length(delindex,:) = [];
    Q(delindex) = [];
    maxParentsQ = max(Q);
    minParentsQ = min(Q);
    minParentsQ_index = find(Q==minParentsQ);
    N_struct = length(DBR_length(:,1));
    
    newDBR_length = zeros(2*nchoosek(N_struct,2),L);
    p = 1;
    for i=1:N_struct-1
        for j=i+1:N_struct
            pc = mod(int16(rand*tens),L); % crossover point
            if pc == 0 % no crossover
                newDBR_length(p,:) = DBR_length(i,:);
                p = p+1;
                newDBR_length(p,:) = DBR_length(j,:);
                p = p+1;
            else
                newDBR_length(p,1:pc-1) = DBR_length(i,1:pc-1);
                newDBR_length(p,pc:end) = DBR_length(j,pc:end);
                p = p+1;
                newDBR_length(p,1:pc-1) = DBR_length(j,1:pc-1);
                newDBR_length(p,pc:end) = DBR_length(i,pc:end);
                p = p+1;
            end
        end
    end
    
    newN_struct = length(newDBR_length(:,1));
    
    % Mutate
    for i=1:newN_struct
        pm = mod(int16(rand*tens),L); % mutant point
        if pm
            a = rand*tens;
            while a == 0
                a = rand*tens;
            end
            newDBR_length(i,pm) = a;
        end
    end
    
    parfor i=1:newN_struct
        newR = calR(DBR_eps,newDBR_length(i,:),lambda,L);
        newQ = calQ(newR,lambda,tarlam);
        if newQ <= maxParentsQ || newR(tarlam_index) < 0.9
            newDBR_length(i,:) = 0;
        end
    end
    
    delindex = find(~newDBR_length(:,1));
    newDBR_length(delindex',:) = [];
    
    if length(newDBR_length(:,1)) == 1
        DBR_length(minParentsQ_index,:) = newDBR_length;
    elseif length(newDBR_length(:,1)) > 1
        DBR_length = newDBR_length;
    end
end

N_struct = length(DBR_length(:,1));
R = zeros(N_struct,length(lambda));
Q = zeros(1,N_struct);
parfor i=1:N_struct
    R(i,:)= calR(DBR_eps,DBR_length(i,:),lambda,L);
    Q(i) = calQ(R(i,:),lambda,tarlam);
end
delindex = find(~Q);
DBR_length(delindex,:) = [];
R(delindex,:) = [];
Q(delindex) = [];
Plot_R(R,lambda);