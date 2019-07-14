function [Q,MSL] = rewardFunc(R,lambda,tarlam_idx)
tarint = R(tarlam_idx);
try
    [Q,MSL] = calReward(R,tarlam_idx,tarint,0.5,lambda);
catch
    
    [Q,MSL] = failReward();
end



function [Q,MSL] = calReward(R,tarlam_idx,tarint,rtar,lambda)
for j=tarlam_idx:1:length(lambda)
    if R(j) <= rtar*tarint
        tarhi = j;
        break;
    end
end

for j=tarlam_idx:-1:1
    if R(j) <= rtar*tarint
        tarlo = j;
        break;
    end
end
if tarlo == tarhi
    [Q,MSL] = failReward();
else
    Q = (1/rtar)*(1/lambda(tarlam_idx))/(1/lambda(tarlo)-1/lambda(tarhi));
    MSL = (rtar)*mean(R(1,[1:tarlo, tarhi:length(lambda)]));
end

function [Q,MSL] = failReward()
Q = -3e-1;
MSL = 1;
