filename = 'alpha_beta.csv';
coords = csvread(filename, 1, 0);
x = coords(:,1);
y = coords(:,2);
J1 = csvread('layer_1/J_1.csv');
J2 = csvread('layer_1/J_2.csv');
J3 = csvread('layer_1/J_3.csv');
J4 = csvread('J4.csv');
J5 = csvread('J5.csv');
J6 = csvread('J6.csv');
J7 = csvread('J7.csv');
J8 = csvread('J8.csv');
J9 = csvread('J9.csv');
J10 = csvread('J10.csv');
J11 = csvread('J11.csv');
J12 = csvread('J12.csv');
J13 = csvread('J13.csv');
J14 = csvread('J14.csv');
J15 = csvread('J15.csv');
J16 = csvread('J16.csv');
J17 = csvread('J17.csv');
J18 = csvread('J18.csv');
J19 = csvread('J19.csv');
J20 = csvread('J20.csv');



x_vec = zeros(length(x)*length(y),1);
y_vec = zeros(length(x)*length(y),1);
J1_vec = zeros(length(x)*length(y),1);
J2_vec = zeros(length(x)*length(y),1);
J3_vec = zeros(length(x)*length(y),1);
J4_vec = zeros(length(x)*length(y),1);
J5_vec = zeros(length(x)*length(y),1);
J6_vec = zeros(length(x)*length(y),1);
J7_vec = zeros(length(x)*length(y),1);
J8_vec = zeros(length(x)*length(y),1);
J9_vec = zeros(length(x)*length(y),1);
J10_vec = zeros(length(x)*length(y),1);
J11_vec = zeros(length(x)*length(y),1);
J12_vec = zeros(length(x)*length(y),1);
J3_vec = zeros(length(x)*length(y),1);
J14_vec = zeros(length(x)*length(y),1);
J15_vec = zeros(length(x)*length(y),1);
J16_vec = zeros(length(x)*length(y),1);
J7_vec = zeros(length(x)*length(y),1);
J18_vec = zeros(length(x)*length(y),1);
J19_vec = zeros(length(x)*length(y),1);
J20_vec = zeros(length(x)*length(y),1);


for i=1:length(x)
    for j=1:length(y)
        index = (i-1)*length(x) + j;
        x_vec(index) = x(i);
        y_vec(index) = y(j);

        J1_vec(index) = J1(i,j);
        J2_vec(index) = J2(i,j);
        J3_vec(index) = J3(i,j);
        J4_vec(index) = J4(i,j);
        J5_vec(index) = J5(i,j);
        J6_vec(index) = J6(i,j);
        J7_vec(index) = J7(i,j);
        J8_vec(index) = J8(i,j);
        J9_vec(index) = J9(i,j);
        J10_vec(index) = J10(i,j);
        J11_vec(index) = J11(i,j);
        J12_vec(index) = J12(i,j);
        J13_vec(index) = J13(i,j);
        J14_vec(index) = J14(i,j);
        J15_vec(index) = J15(i,j);
        J16_vec(index) = J16(i,j);
        J17_vec(index) = J17(i,j);
        J18_vec(index) = J18(i,j);
        J19_vec(index) = J19(i,j);
        J20_vec(index) = J20(i,j);
    end
end
      
% interpolation grid
xq = 0:0.01:2.0;
yq = 0.:0.01:2.0;

% unstack
x_new = zeros(length(xq)*length(yq),1);
y_new = zeros(length(xq)*length(yq),1);
for i=1:length(xq)
    for j=1:length(yq)
        index = (i-1)*length(xq)+j;
        x_new(index) = xq(i);
        y_new(index) = yq(j);
    end
end

% compute surface
J1q = griddata(x_vec,y_vec,J1_vec,x_new,y_new, 'cubic');
J2q = griddata(x_vec,y_vec,J2_vec,x_new,y_new, 'cubic');
J3q = griddata(x_vec,y_vec,J3_vec,x_new,y_new, 'cubic');
J4q = griddata(x_vec,y_vec,J4_vec,x_new,y_new, 'cubic');
J5q = griddata(x_vec,y_vec,J5_vec,x_new,y_new, 'cubic');
J6q = griddata(x_vec,y_vec,J6_vec,x_new,y_new, 'cubic');
J7q = griddata(x_vec,y_vec,J7_vec,x_new,y_new, 'cubic');
J8q = griddata(x_vec,y_vec,J8_vec,x_new,y_new, 'cubic');
J9q = griddata(x_vec,y_vec,J9_vec,x_new,y_new, 'cubic');
J10q = griddata(x_vec,y_vec,J10_vec,x_new,y_new, 'cubic');
J11q = griddata(x_vec,y_vec,J11_vec,x_new,y_new, 'cubic');
J12q = griddata(x_vec,y_vec,J12_vec,x_new,y_new, 'cubic');
J13q = griddata(x_vec,y_vec,J13_vec,x_new,y_new, 'cubic');
J14q = griddata(x_vec,y_vec,J14_vec,x_new,y_new, 'cubic');
J15q = griddata(x_vec,y_vec,J15_vec,x_new,y_new, 'cubic');
J16q = griddata(x_vec,y_vec,J16_vec,x_new,y_new, 'cubic');
J17q = griddata(x_vec,y_vec,J17_vec,x_new,y_new, 'cubic');
J18q = griddata(x_vec,y_vec,J18_vec,x_new,y_new, 'cubic');
J19q = griddata(x_vec,y_vec,J19_vec,x_new,y_new, 'cubic');
J20q = griddata(x_vec,y_vec,J20_vec,x_new,y_new, 'cubic');


L = sqrt(size(J1q,1));
J1_new = reshape(J1q, [L,L])'; 
J2_new = reshape(J2q, [L,L])'; 
J3_new = reshape(J3q, [L,L])'; 
J4_new = reshape(J4q, [L,L])'; 
J5_new = reshape(J5q, [L,L])'; 
J6_new = reshape(J6q, [L,L])'; 
J7_new = reshape(J7q, [L,L])'; 
J8_new = reshape(J8q, [L,L])'; 
J9_new = reshape(J9q, [L,L])'; 
J10_new = reshape(J10q, [L,L])'; 
J11_new = reshape(J11q, [L,L])'; 
J12_new = reshape(J12q, [L,L])'; 
J13_new = reshape(J13q, [L,L])'; 
J14_new = reshape(J14q, [L,L])'; 
J15_new = reshape(J15q, [L,L])'; 
J16_new = reshape(J16q, [L,L])'; 
J17_new = reshape(J17q, [L,L])'; 
J18_new = reshape(J18q, [L,L])'; 
J19_new = reshape(J19q, [L,L])'; 
J20_new = reshape(J20q, [L,L])'; 
%%
[X,Y] = meshgrid(xq,yq);
tot = X + Y;
for i=1:L
    for j=1:L
        if tot(i,j) > 2.0 || tot(i,j) < 0.0
            J1_new(i,j) = NaN;
            J2_new(i,j) = NaN;
            J3_new(i,j) = NaN;
            J4_new(i,j) = NaN;
            J5_new(i,j) = NaN;
            J6_new(i,j) = NaN;
            J7_new(i,j) = NaN;
            J8_new(i,j) = NaN;
            J9_new(i,j) = NaN;
            J10_new(i,j) = NaN;
            J11_new(i,j) = NaN;
            J12_new(i,j) = NaN;
            J13_new(i,j) = NaN;
            J14_new(i,j) = NaN;
            J15_new(i,j) = NaN;
            J16_new(i,j) = NaN;
            J17_new(i,j) = NaN;
            J18_new(i,j) = NaN;
            J19_new(i,j) = NaN;
            J20_new(i,j) = NaN;
        end
    end
end

%% plot 1
figure
surfc(X,Y,J1_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J1(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J1(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J1(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J1(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J1(1,end)-0.02,'Coajoint');
text(X(end,1),Y(end,1),J1(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
%zlim([0.61, 0.85])
%xlim([0,1.05])
%ylim([0,1.05])
hold off

% plot 2
figure
surfc(X,Y,J2_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J2(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J2(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J2(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J2(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J2(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J2(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

% plot 3
figure
surfc(X,Y,J3_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J3(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J3(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J3(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J3(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J3(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J3(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

% plot 4
figure
surfc(X,Y,J4_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J4(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J4(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J4(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J4(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J4(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J4(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

% plot 5
figure
surfc(X,Y,J5_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J5(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J5(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J5(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J5(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J5(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J5(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

% plot 6
figure
surfc(X,Y,J6_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J6(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J6(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J6(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J6(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J6(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J6(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

% plot 7
figure
surfc(X,Y,J7_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J7(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J7(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J7(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J7(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J7(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J7(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

% plot 8
figure
surfc(X,Y,J8_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J8(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J8(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J8(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J8(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J8(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J8(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

% plot 9
figure
surfc(X,Y,J9_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J9(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J9(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J9(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J9(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J9(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J9(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

% plot 10
figure
surfc(X,Y,J10_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J10(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J10(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J10(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J10(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J10(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J10(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

% plot 11
figure
surfc(X,Y,J11_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J11(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J11(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J11(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J11(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J11(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J11(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

% plot 12
figure
surfc(X,Y,J12_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J12(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J12(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J12(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J12(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J12(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J12(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

% plot 13
figure
surfc(X,Y,J13_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J13(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J13(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J13(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J13(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J13(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J13(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

% plot 14
figure
surfc(X,Y,J14_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J14(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J14(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J14(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J14(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J14(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J14(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

% plot 15
figure
surfc(X,Y,J15_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J15(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J15(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J15(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J15(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J15(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J15(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

% plot 16
figure
surfc(X,Y,J16_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J16(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J16(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J16(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J16(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J16(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J16(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

% plot 17
figure
surfc(X,Y,J17_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J17(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J17(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J17(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J17(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J17(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J17(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

% plot 18
figure
surfc(X,Y,J18_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J18(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J18(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J18(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J18(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J18(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J18(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

% plot 19
figure
surfc(X,Y,J19_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J19(1,1),'.r','markersize',25);
plot3(X(1,end), Y(1,end), J19(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J19(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J19(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J19(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J19(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

% plot 20
figure
surfc(X,Y,J20_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J20(1,1),'.r','markersize',25);
plot3(X(1,end), Y(1,end), J20(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J20(end,1),'.r','markersize',25);

% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J20(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J20(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J20(end,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

%%
% plot 1
figure
surfc(X,Y,J1_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J1(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J1(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J1(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J1(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J1(1,end)-0.02,'Coajoint');
text(X(end,1),Y(end,1),J1(end,1)-0.02,'Adjoint');
title("Toy Example (L1): Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
%zlim([0.61, 0.85])
%xlim([0,1.05])
%ylim([0,1.05])
hold off

% plot 2
figure
surfc(X,Y,J2_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J2(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J2(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J2(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J2(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J2(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J2(end,1)-0.02,'Adjoint');
title("Toy Example (L1): Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off

% plot 3
figure
surfc(X,Y,J3_new,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(X(1,1), Y(1,1), J3(1,1),'.r','markersize',26);
plot3(X(1,end), Y(1,end), J3(1,end),'.r','markersize',25);
plot3(X(end,1), Y(end,1), J3(end,1),'.r','markersize',25);
% label init, adjoint and coadoint
text(X(1,1),Y(1,1),J3(1,1)-0.02,'Init');
text(X(1,end),Y(1,end),J3(1,end)-0.02,'Coadjoint');
text(X(end,1),Y(end,1),J3(end,1)-0.02,'Adjoint');
title("Toy Example (L1): Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off
