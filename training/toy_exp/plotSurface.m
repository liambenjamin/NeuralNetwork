% Set directory to file location

filename = 'alpha_beta_21.csv';
coords = csvread(filename, 1, 0);
x = coords(:,1);
y = coords(:,2);
J1 = csvread('simplexF21_4_1.csv');
J2 = csvread('simplexF21_4_2.csv');


x_vec = zeros(length(x)*length(y),1);
y_vec = zeros(length(x)*length(y),1);
J1_vec = zeros(length(x)*length(y),1);
J2_vec = zeros(length(x)*length(y),1);

for i=1:length(x)
    for j=1:length(y)
        index = (i-1)*length(x) + j;
        x_vec(index) = x(i);
        y_vec(index) = y(j);

        J1_vec(index) = J1(i,j);
        J2_vec(index) = J2(i,j);
    end
end
      
% interpolation grid
xq = 0:0.02:2.0;
yq = 0.:0.02:2.0;

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

L = sqrt(size(J1q,1));
J1_new = reshape(J1q, [L,L])'; 
J2_new = reshape(J2q, [L,L])'; 
%
[X,Y] = meshgrid(xq,yq);
for i=1:L
    for j=1:L
        if i > L-j+1
            J1_new(i,j) = NaN;
            J2_new(i,j) = NaN;
        end
    end
end

J1 = J1_new;
J2 = J2_new;
%% plot 1
figure
surfc(X,Y,J1,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(0, 0, J1(1,1),'.r','markersize',25); %init
plot3(1, 0, J1(1,51),'.r','markersize',25); %coadjoint
plot3(0, 1, J1(51,1),'.r','markersize',25); %adjoint
% label init, adjoint and coadoint
text(0,0,J1(1,1)-0.02,'Init');
text(1,0,J1(1,51)-0.02,'Coadjoint');
text(0,1,J1(51,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
%zlim([0.61, 0.85])
%xlim([0,1.05])
%ylim([0,1.05])
hold off

%plot 2
figure
surfc(X,Y,J2,'FaceAlpha',0.7);
hold on
colorbar
% highlight init, adjoint and coadjoint solutions
plot3(0, 0, J2(1,1),'.r','markersize',25); %init
plot3(1, 0, J2(1,51),'.r','markersize',25); %coadjoint
plot3(0, 1, J2(51,1),'.r','markersize',25); %adjoint
% label init, adjoint and coadoint
text(0,0,J2(1,1)-0.02,'Init');
text(1,0,J2(1,51)-0.02,'Coadjoint');
text(0,1,J2(51,1)-0.02,'Adjoint');
title("Loss Surface Interpolation over 2-Simplex");
xlabel("\alpha");
set(get(gca,'zlabel'),'rotation',0)
ylabel("\beta");
zlabel("J(\theta)        ");
hold off
