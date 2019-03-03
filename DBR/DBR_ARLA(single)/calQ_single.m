function [Q, MSL] = calQ_single(R,lambda,tarlam_index)
tarint = single(R(tarlam_index));
if tarint < 0.9
    Q = 0;
    MSL = 1;
else
    try
        for j=tarlam_index:1:length(lambda)
            if R(j) <= 0.5*tarint
                tarhi = j;
                break;
            end
        end
        
        for j=tarlam_index:-1:1
            if R(j) <= 0.5*tarint
                tarlo = j;
                break;
            end
        end
        Q = tarint/(lambda(tarhi)-lambda(tarlo));
        MSL = mean(R([1:tarlo, tarhi:end]));
    catch
        Q = 0;
        MSL = 1;
    end
end