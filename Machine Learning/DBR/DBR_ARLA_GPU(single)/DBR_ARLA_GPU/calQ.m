function [Q, MSL] = calQ(R,lambda,Nstruct,tarlam_index)
Q = ones(Nstruct,1);
MSL = ones(Nstruct,1);

for i=1:Nstruct
    tarint = R(i,tarlam_index);
    if tarint < 0.9
        Q(i) = 0;
        MSL(i) = 1;
    else
        try
            for j=tarlam_index:1:length(lambda)
                if R(i,j) <= 0.5*tarint
                    tarhi = j;
                    break;
                end
            end
            
            for j=tarlam_index:-1:1
                if R(i,j) <= 0.5*tarint
                    tarlo = j;
                    break;
                end
            end
            Q(i) = tarint/(lambda(tarhi)-lambda(tarlo));
            MSL(i) = mean(R([1:tarlo, tarhi:end]));
        catch
            Q(i) = 0;
            MSL(i) = 1;
        end
    end
end