%% ---- Eckert's Reference Temperature, Cone Example ----

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

%% ---- HEAT TRANSFER FOR A WEDGE, MACH 15, 40km ALTITUDE ----

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

%% ---- FUNCTIONS BELOW HERE ---- 

%% Oblique Perfect Gas Function 

function [H2, V2, T2, P2, rho2,beta_deg,M2,a2] = perfgas_oblique(M,V1,T1,P1,rho_1,little_del_w,gamma,R_specific,cp1,a1,theta)
%For our initial guess at beta, let's use the M>>1 approximation. Although
%our flow M>>1, this code will calculate a more accurate turn angle than the approximation. 
b_init = ((gamma + 1) / 2)*theta;

%We'll use Newton's method, which requires the function and the function's 
%derivative, included below as "f" and "fp"
%  tan(theta) = 2cot(beta)*(M^2sin^2(beta) - 1)/(M^2(gamma + cos(2beta) + 1)

b = zeros(10,1);
i = 1;
b(i) = b_init;
for i = 1:10
    b(i+1) = b(i) - ((2*cot(b(i))*(M^2*(sin(b(i)))^2-1))/(M^2*(gamma + cos(2*b(i)))+2)-tan(theta))...
        / ((4*M^2*sin(2*b(i))*cot(b(i))*(M^2*sin(b(i))^2-1))/( (M^2*(cos(2*b(i))+gamma)+2) ^2)...  
                + (4*M^2*cos(b(i))^2 - 2*csc(b(i))^2*(M^2*(sin(b(i)))^2-1))/(M.^2.*(cos(2*b(i))+gamma)+2));
end
beta = b(10);
beta_deg = rad2deg(b(10));

%First, let's calculate the Mach number at 2
M1 = M;
M2 = sqrt(((1+((gamma-1)/2)*(M1^2)*(sin(beta)^2)) / (gamma*(M1^2)*(sin(beta)^2)-((gamma-1)/2)))...
    * (1/(sin(beta-theta)^2)));
m_ratio = M2/M1;

temp_ratio = 1 + ((2*(gamma-1))/((gamma+1)^2)) * (((M1^2)* (sin(beta)^2) - 1) / ((M1^2)* (sin(beta)^2)))...
    * (gamma*(M1^2)*(sin(beta)^2)+1);
T2 = temp_ratio*T1;

H2 = cp1*T2*1000;

%Using the relation T2/T1 = (a2/a1)^2, we can also solve for the ratio of
%a2/a1 
a_ratio = sqrt(temp_ratio);
a2 = a_ratio*a1;

rho_ratio = ( (gamma+1)*(M1^2)*(sin(beta)^2) ) / ( (gamma-1)*(M1^2)*(sin(beta)^2) + 2 );
rho2 = rho_ratio*rho_1;

v_ratio = m_ratio* a_ratio;
V2 = v_ratio*V1;

p_ratio = rho_ratio*temp_ratio;
P2 = p_ratio*P1;

T01 = T1*(1+ ((gamma-1)/2)*(M1^2));
T02 = T2*(1+ ((gamma-1)/2)*(M2^2));

total_p_ratio = p_ratio * (( 1 + ((gamma-1)/2)*(M2^2) ) ^(gamma/(gamma-1)))...
    / (( 1 + ((gamma-1)/2)*(M1^2) ) ^(gamma/(gamma-1)));
P01 = P1* ((1 +((gamma-1)/2)*(M1^2))^((gamma)/(gamma-1)));
P02 = P01*total_p_ratio;
end
%% Alternative Mu calc
function mu = mu_dvg(T)
mu_ref = 1.8e-5;
T_ref = 300; %K
mu = mu_ref*((T/T_ref)^0.7);
end
%% Sutherland's Viscosity Calc

function mu = mu_suth(T)
b = 1.4685e-6;
S = 110.4;

mu = b*(T^(3/2)) / (T+S);
end