function Q = calQ(R,lambda,tarlam)
tarlam_index = find(lambda==tarlam);
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
    Q = tarlam/(lambda(tarhi)-lambda(tarlo));
catch
    Q = 0;
end