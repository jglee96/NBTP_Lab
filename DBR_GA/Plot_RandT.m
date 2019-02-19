function Plot_RandT(R,T,lambda)
N = length(R(:,1));
% figure
% for i=1:N
%     subplot(N,1,i);
%     plot(lambda, T(i,:));
% end

figure
for i=1:N
    subplot(N,1,i);
    plot(lambda, R(i,:));
end