function Plot_R(R,lambda)
N = length(R(:,1));

figure
for i=1:N
    subplot(N,1,i);
    plot(lambda, R(i,:));
end