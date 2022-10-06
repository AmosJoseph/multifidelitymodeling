clc;
close all;
clear all;
%
%%%%%%%%%%%%%%%
% User Inputs %
%%%%%%%%%%%%%%%
S = 5000;       % generate 5000 samples
P = 6;          % 6 parameters (6 modes)
Type = 'S';     % Use Sobol set
Sphere = 1:P;   % Use spherical coordinates for modal weights
Z = 1;          % Include zero deformation
mult_tr = 1.05; % Increase bounds by 5%
mult_te = 0.95; % Decrease bounds by 5% 
%
% Bounds on modal weights
mB = [1e-2,1e-3,1e-4,1e-4,1e-4,1e-5];
%
% Estimated sample points for surr and test
pts_surr = 200;
pts_test = 50;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create Parameter Space %
%%%%%%%%%%%%%%%%%%%%%%%%%%
Bound_tr = NaN(P,2);
Bound_te = NaN(P,2);
cnt = 1;
for i = 1:P
    Bound_tr(i,:) = 2.*[-mB(cnt);mB(cnt)].*mult_tr;
    Bound_te(i,:) = 2.*[-mB(cnt);mB(cnt)].*mult_te;
    cnt = cnt + 1;
end

Bound = Bound_tr;
rand_seed = 1291874;
zpts = 1;
%
% Adjusts clustering of points for hypersphere
cluster = 2.5; %2.5;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check for Input Variables %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isnan(Sphere)
    HS = 'n';
else
    HS = 'y';
end
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct Sobol/Halton Set %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (Type == 'H')
    Set = haltonset(P+1,'Leap',rand_seed);
    Set = scramble(Set,'MatousekAffineOwen');
elseif (Type == 'S')
    Set = sobolset(P+1,'Leap',rand_seed);
    Set = scramble(Set,'MatousekAffineOwen');
else
    display(':: Incorrect Type ::');
    return;
end
%
X0 = net(Set,S);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set Hypercube Parameters %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Y = NaN(S,P);
for i = 1:P
    Y(:,i) = X0(:,i).*(Bound(i,2)-Bound(i,1)) + Bound(i,1);
end

%%

pts = length(Sphere);
s = (X0(:,Sphere(1):Sphere(end)).*2 - 1).*1;

%%
U = X0(:,end).^cluster;
%%
sumsq = sum(s.^2,2);
%%
for i = 1:pts
    j = Sphere(i);
    Y(:,j) = s(:,i).*(U.^(1/3)).*Bound(j,2)./sqrt(sumsq);
end