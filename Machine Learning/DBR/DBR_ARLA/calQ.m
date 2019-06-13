function [Q, MSL] = calQ(R,lambda,tarlam_index)
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
    Q = tarint/(lambda(tarhi)-lambda(tarlo));
    MSL = mean(R([1:tarlo, tarhi:end]));
catch
    Q = 0;
    MSL = 1;
end