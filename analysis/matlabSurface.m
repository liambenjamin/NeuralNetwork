%% Loading Data
mesh = csvread('toy/mesh.csv');
F = csvread('toy/F.csv');
G = csvread('toy/G.csv');
E = csvread('toy/E.csv');
Fr1 = csvread('toy/Fr1.csv');
Gr1 = csvread('toy/Gr1.csv');
Er1 = csvread('toy/Er1.csv');
Fr2 = csvread('toy/Fr2.csv');
Gr2 = csvread('toy/Gr2.csv');
Er2 = csvread('toy/Er2.csv');
Fr3 = csvread('toy/Fr3.csv');
Gr3 = csvread('toy/Gr3.csv');
Er3 = csvread('toy/Er3.csv');
alpha = mesh(:,1);
beta = mesh(:,2);

%% Plotting Data
surf(alpha,beta,F);
title("Four Point Interpolation");
xlabel("alpha");
ylabel("beta");
zlabel("Cost");
text(alpha(1),beta(1),F(1,1)-0.01,'Adjoint');
text(alpha(1),beta(21),F(21,1)+0.01,'Coadjoint(ntwk2)');
text(alpha(21),beta(1),F(1,21)-0.01,'Coadjoint(ntwk3)');
text(alpha(21),beta(21),F(21,21)+0.01,'Random Pt');

%% Plotting w/ Random Initialization
surf(alpha,beta,Fr1);
title("e1 Two Random Four Point Interpolation");
xlabel("alpha");
ylabel("beta");
zlabel("Cost");
text(alpha(1),beta(1),F(1,1)-0.01,'Adjoint');
text(alpha(1),beta(21),F(21,1)+0.01,'Rand Ntwk 1');
text(alpha(21),beta(1),F(1,21)-0.01,'Rand Ntwk 2');
text(alpha(21),beta(21),F(21,21)+0.01,'Coadjoint 1e1');

%% Plotting w/ Random Initialization
surf(alpha,beta,Fr2);
title("e2 Two Random Four Point Interpolation");
xlabel("alpha");
ylabel("beta");
zlabel("Cost");
text(alpha(1),beta(21),F(21,1)+0.01,'Rand Ntwk 1');
text(alpha(21),beta(1),F(1,21)-0.01,'Rand Ntwk 2');
text(alpha(21),beta(21),F(21,21)+0.01,'Coadjoint 1e2');

%% Plotting w/ Random Initialization
surf(alpha,beta,Fr3);
title("e3 Two Random Four Point Interpolation");
xlabel("alpha");
ylabel("beta");
zlabel("Cost");
text(alpha(1),beta(1),F(1,1)-0.01,'Adjoint');
text(alpha(1),beta(21),F(21,1)+0.01,'Rand Ntwk 1');
text(alpha(21),beta(1),F(1,21)-0.01,'Rand Ntwk 2');
text(alpha(21),beta(21),F(21,21)+0.01,'Coadjoint 1e3');

%% Plotting w/ Random Initialization

figure
surf(alpha, beta, Fr1)
title("e3 Two Random Four Point Interpolation");
xlabel("alpha");
ylabel("beta");
zlabel("Cost");
text(alpha(1),beta(1),F(1,1)-0.01,'Adjoint');
text(alpha(1),beta(21),F(21,1)+0.01,'Rand Ntwk 1');
text(alpha(21),beta(1),F(1,21)-0.01,'Rand Ntwk 2');
text(alpha(21),beta(21),F(21,21)+0.01,'Coadjoint 1e3');
hold on
surf(alpha, beta, Fr2)
surf(alpha, beta, Fr3)
hold off
grid on
