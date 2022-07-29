%% Eckert's Reference Temperature, Cone

close all
clear variables
clc

% RUN #0
gamma = 1.4; % perfect gas
T1 = 227.130; %Kelvin
T_w = 275; %Kelvin
u_inf = 

% a1 = 302.122; %m/s
% u1 = 3657.6; % m/s
% M1 = u1 / a1; %dimensionless


P1 = 1090.16; %Pascals
rho1 = 0.0167207; %kg/m^3

P0_1 = P1 * (( 1+ ((gamma-1)/2)*(M1^2) )^(gamma/(gamma-1))) ;

%% Calculate post-shock conditions with perfect gas assumptions
M = 25; %Mach number
theta = deg2rad(25); %turn angle
gamma = 1.4;  
little_del_w = theta; 
R_specific = 287.058; %J/Kg*K
T1 = 250; %HW2, Problem 4, Kelvin
P1 = 17.8; %HW2, Problem 4, Pascals
V1 = 4764.05;
rho_1 = .0039;
cp1 = 1.005; %KJ/Kg*K, air at 251K
T_W = 800;
a1 = sqrt(gamma*R_specific*T1);
% a1 = 312.3;
%M1 = 15;
[H2, V2, T2, P2, rho2,beta_deg,M2,a2] = perfgas_oblique(M,V1,T1,P1,rho_1,little_del_w,gamma,R_specific,cp1,a1,theta);

%% Calculate perfect gas heat transfer as a function of distance 
chrex = 0.38;
x_loc = 4; %in meters

mu = mu_dvg(T2);

% heat transfer calculations
Pr = .715;
r = sqrt(Pr);

h_w = (gamma/(gamma-1))*R_specific*T_W;
h_0 = H2 + .5*(V2^2);
h_aw = H2 + r*(h_0 - H2);
cp = (gamma*R_specific/(gamma-1));
T_0 = T2*(1+ ((gamma-1)/2)*(M2^2));
T_AW = r*(T_0 - T2) + T2;
x = 0:.001:10;
Q_wall = (chrex*((mu./x).^.5))*(h_aw-h_w)*sqrt(rho2*V2);
% Q_wall = (chrex*((mu./x).^.5))*cp*(T_AW - T_W)*sqrt(rho2*V2);

%Picking a specfic point on the wall
[minValue,closestIndex] = min(abs(x-x_loc));
HeatTransferDesiredPoint = Q_wall(closestIndex)/10000;
disp(HeatTransferDesiredPoint)

semilogy(x,Q_wall)
grid on
xlim([-.1 10.1])
xlabel('Location Along Wedge Wall, $x$ (meters)','FontSize',15,'FontWeight','Bold','Interpreter','Latex')
ylabel('Heat Transfer Rate ($q_w$)','FontSize',15,'FontWeight','Bold','Interpreter','Latex')
% filename = 'Part2_Heat_Transfer.eps';
% saveas(gcf,filename, 'epsc')
% legend(leg,'location','northeast','interpreter','latex','FontSize',11)