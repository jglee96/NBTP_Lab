%This program will iterate through a sequence of starting points - randomly picked - then generate a file like the one I have for the excel doc.
% This program will design to a given configuration.

addpath 'spherical_T_matrix';
addpath 'spherical_T_matrix/bessel';

lamLimit = 400

lambda = linspace(lamLimit, 800, (800-lamLimit)+1)';
omega = 2*pi./lambda;

eps_silver = interp1(data.omega_silver,data.epsilon_silver,omega);
eps_silica = 2.04*ones(length(omega), 1);
my_lam = lambda./1000;
eps_tio2 = 5.913+(.2441)*1./(my_lam.*my_lam-.0803);
eps_water  = 1.77*ones(length(omega), 1);


%%%%% =========================
%%%%% HyperParameters to Choose

% Manually pick your layers.
% Example for 5 layer 
eps  = [eps_silica eps_tio2 eps_silica eps_tio2 eps_silica eps_water]
%eps = [eps_silica eps_silver eps_silica eps_silver eps_silica eps_water];
%eps = [eps_silver eps_tio2 eps_silica eps_tio2 eps_silica eps_water];
network_file = 'results/5_layer_tio2/'
numberOpts = 10
useStoredStartParams = true
useNN = true
useGradient = false
lowDesirePoint = 40 % 400+ 2*n
highDesirePoint = 45
lowThick = 30
highThick = 70

if useStoredStartParams == false 
	all_start_params = []
	eps_size = size(eps);
	eps_size = eps_size(2);
	for i = 1:numberOpts
		start_params = []
	    for j = 1:(eps_size-1)
	        j
	    	start_params = [start_params ; round(rand*(highThick-lowThick)+lowThick,1)];
	    end
	    all_start_params = [all_start_params , start_params];
	end
	order = 25;
	if length(start_params) ==2 || length(start_params) == 3
		order = 4;
	end
	if length(start_params)  == 4 || length(start_params) == 5
		order = 9;
	end
	if length(start_params) == 6 || length(start_params) == 7
		order = 12;
	end
	if length(start_params) == 8 || length(start_params) == 9
		order = 15;
	end
	if length(start_params) == 10 || length(start_params) == 11
	    order = 18;
	end
	%order = 3;
end
order = 10;

wgts = cell(0);
bias = cell(0);
for i=0:4
    wgts{i+1} = transpose(load(strcat(network_file,'w_',num2str(i),'.txt')));
    bias{i+1} = load(strcat(network_file,'b_',num2str(i),'.txt'));
end
dim = size(wgts);

%Get the spect file. 
filename2 = strcat(network_file,'spec_file_0.txt');
myspect2 = csvread(filename2);
means = transpose(myspect2(1,:));
stds = transpose(myspect2(2,:));

options = optimoptions('fmincon','Display','iter','Algorithm','interior-point','ObjectiveLimit',.01,'SpecifyObjectiveGradient',useGradient);

if useNN == false
	cost_func_nn = @(x)cost_function_math_desired(x,wgts,bias,dim(2),lowDesirePoint,highDesirePoint,omega,eps,order,lambda);
end
if useNN == true
	cost_func_nn = @(x)cost_function_nn_desired(x,wgts,bias,dim(2),lowDesirePoint,highDesirePoint,means,stds,lambda);
end
%This is the actual computation
totconv = 0;
tottime = 0;
convergence_best = 1000.0;
yval = 0;
for i = 1:numberOpts
	start_params = all_start_params(:,i)
  %start_params = [10;47;27;36;10]

	[mytime, convergence,x] = run_opt(start_params,cost_func_nn,options,lowThick,highThick);
	if convergence< convergence_best
		convergence_best = convergence
		myval = x;
	end
end
convergence_best
x

% Now graph the results
hold on
spect = scatter_sim_0_gen_spect_faster(x,omega,eps,order,lambda)./(3*lambda.*lambda)*2*pi;
%spect = scatter_sim_0_gen_single_spect(x).*(pi*sum(x)^2)./(3*lambda.*lambda)*2*pi;
area([lambda(lowDesirePoint*2),lambda(highDesirePoint*2)],[max(spect(1:2:(800-lamLimit)+1,1)),max(spect(1:2:(800-lamLimit)+1,1))],'EdgeColor','none')
alpha(.2)
plot(lambda(1:2:(800-lamLimit)+1),[spect(1:2:(800-lamLimit)+1,1)])
hold off

xlabel('Wavelength (nm)');
ylabel('\sigma/\pi r^2');
title('Geometries to match desired spectrums');
legend('Desired scattering',strcat('NN - Nanoparticle'));%,'a');



function [time,convergence,x] = run_opt(start_params,cost_func,options,lowThick,highThick)
A = [];
b = [];
Aeq = [];
beq = [];
lb = lowThick * ones(1,length(start_params));
ub = highThick * ones(1,length(start_params));
nonlcon=[];
x0 = start_params;
tic;
[x,fval,exitflag,output] = fmincon(cost_func,x0,A,b,Aeq,beq,lb,ub,nonlcon, options);
x
time = toc;
convergence = fval;
end

function [cost,gradient] = cost_function_nn_desired(x,weights,biases,depth,desiredLowVal,desiredUpVal,xmeans,xstds,lambda)
  % how big the spectrum is.
  N=201;
  vec=zeros(N,1);
  positions=[desiredLowVal:desiredUpVal];
  vec(positions)=1; 
  r = x;

  x = (x-xmeans)./xstds;

  %This is thus the multiplicative thing 
  [layer, Jacobian] = NN(weights,biases,x);
  % If you want to use the transposed thing.
  lambda = lambda(1:2:length(lambda));
  %Uncomment the last part to remove normalization by size. 
  layer = layer./(3*lambda.*lambda)*2*pi.*(pi*sum(r)^2);

  %size(layer)
  %size(vec)
  topVal = mean(layer.*abs(1-vec));
  botVal = mean(layer.*vec);
  cost = topVal/botVal;
  % Thus should be Xo-Xd.
  % error per point should be the layer value for all points NOT in the region
  % Positi  ve for all points NOT in the region, negative for all points in the region
  scalingFactor = layer.*abs(1-vec)-layer.*vec;
  %scalingFactor = vec + abs(1-vec)./topVal;
  gradient = transpose(Jacobian) * scalingFactor.*cost;%;.*scalingFactor .*cost;
end

function [cost,gradient] = cost_function_math_desired(x,weights,biases,depth,desiredLowVal,desiredUpVal,omega,eps,order,lambda)
	%input = x;
	%layer = max(0,weights{1}*input)+biases{1};
  %for j=2:depth-1;
  %  	layer = max(0,weights{j}*layer)+biases{j};
  %end
  N=round((800-length(lambda))/2.0)+1.0;
  vec=zeros(N,1);
  positions=[desiredLowVal:desiredUpVal];
  vec(positions)=1; 
  %This is thus the multiplicative thing 
  [layer, Jacobian] = NN(weights,biases,x);

  spectrum_run = scatter_sim_0_gen_spect_faster(x,omega,eps,order,lambda);
  spectrum_new = spectrum_run(1:2:(800-length(lambda))+3,1);

  %layer = weights{depth}*layer+biases{depth};
  %Jacobian = layer;
  %cost = sum(layer)./(layer(49)+layer(50)+layer(51));
  %cost = mean(layer)/mean(layer(50:60,:));
  %length(spectrum_new)
  topVal = mean(spectrum_new);
  botVal = mean(spectrum_new.*vec);
  cost = topVal/botVal;
                  %These live when it is a value
  scalingFactor = 1.0;%vec + abs(1-vec)./topVal;

  %cost = mean(layer)/mean(layer(desiredLowVal:desiredUpVal,:));
  %The gradient is slightly weird, but effectively it is 
  %Similar to before, I need a list of 0's and 1's, and we can go from there.
  %saclingFactor = 
  gradient = transpose(Jacobian)*scalingFactor;
  %gradient = Jacobian2Gradient(Jacobian,layer,spectCompare)*2.0;
end

function spectrum = scatter_sim_0_gen_spect_faster(r,omega,eps,order,lambda)
  % This optimizes with respect to the normalized
  spectrum = total_cs(r,omega,eps,order)./(3*lambda.*lambda)*2*pi;
  % This overall
  %spectrum = total_cs(r,omega,eps,order)/(pi*sum(r)^2);
end


