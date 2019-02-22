function [Layer,R,Q,MSL] = DelDBR(Layer,R,Q,MSL)
delindex = find(~Q);
Layer(delindex,:) = [];
R(delindex,:) = [];
Q(delindex) = [];
MSL(delindex) = [];
