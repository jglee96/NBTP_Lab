function [Q,MSL] = rewardFunc(R,lambda,tarlam_index)
tarint = R(tarlam_index);
j = tarlam_index;

try
    while 1
        if R(j) <= 0.5*tarint
            tarhi = j;
            break;
        else
            j=j+1;
        end
    end
    j = tarlam_index;
    while 1
        if R(j) <= 0.5*tarint
            tarlo = j;
            break;
        else
            j=j-1;
        end
    end
    if tarlo == tarhi
        Q = 0;
        MSL = 1;
    else
        Q = lambda(tarlam_index)/(1/lambda(tarlo)-1/lambda(tarhi));
        MSL = mean(R([1:tarlo, tarhi:end]));
    end
catch
    Q = 0;
    MSL = 1;
end