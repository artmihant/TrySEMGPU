% Spectral Element Method (SEM) implemented for 2D elastic wave equation
% div(σ(grad(dU))) = ρ(∂^2(dU)/∂t^2+damp*∂dU/∂t) - incremental formulation
% dU - differential displacements on the load step
% on generally unstructured curvilinear quadrangular mesh
clear
close all
%%%%%%%%%%%%%%%%%%%%PARAMETERS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
matlab_run = 1
N = 15; %order of SEM
Lx = 4; %size of the domain in X
Ly = 4; %size of the domain in Y
r_i = min(Lx,Ly)/20; %radius of inclusion 
c_i = [0; -Ly/2]; %inclusion's central point

%domain's mechanical properties
K_d = 2e10; %bulk modulus
G_d = 1e10; %shear modulus
coh_d = 3e7; %cohesion
phi_d = pi/6; %internal friction angle
psi_d = pi/18; %dilatancy angle
rho_d = 2000;%density
[Vp_d, Vs_d] = Velocity(K_d, G_d, rho_d);

%%%%%%%%%%%%%%%%%%MESHING%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[r_e, SEM_Nodes, SEM_Elements, SEM_Material] = CircleMesh(N, r_i, 23, 23);
%[SEM_Nodes, SEM_Elements, SEM_Material] = SquareMesh(N, r_i, Lx, Ly, r_i*3, false);
node_size = length(SEM_Nodes(1,:));
elem_size = length(SEM_Material);
[xL, wL, dM] = GLL(N); %initialize SEM data (GLL-quadrature and shape functions' derivatives)
damp = 4*Vp_d/r_e; %damping parameter
max_Time = 30*r_e/Vs_d; %maximum time depends on the shear wave speed
max_Def = 1.e-2; %maximal deformation
max_step = 1.e-4; %maximal deformation per load step
T_vis = 100; %discretization step for visualization

%%%%%%%%%%%%%%%%%%%%PREPROCESSING%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = SEM_Nodes(1,:); %x-coordinates of SEM nodes
Y = SEM_Nodes(2,:); %y-coordinates of SEM nodes

un = zeros(2, node_size); %list of full Lagrangian displacements in SEM nodes
dun = zeros(2, node_size); %list of differential displacements in SEM nodes
dvn = zeros(2, node_size); %list of differential velocities in SEM nodes
dan = zeros(2, node_size); %list of differential accelerations in SEM nodes

%derivatives of the mapping from reference to physical coordinates
dXdKsi = zeros([elem_size,size(dM)]);
dYdKsi = zeros([elem_size,size(dM)]);
dXdEta = zeros([elem_size,size(dM)]);
dYdEta = zeros([elem_size,size(dM)]);
%displacements' derivatives
dUxdKsi = zeros([elem_size,size(dM)]);
dUydKsi = zeros([elem_size,size(dM)]);
dUxdEta = zeros([elem_size,size(dM)]);
dUydEta = zeros([elem_size,size(dM)]);
%accumulated plastic Cauchy-Green strain tensor
epsP_xx = zeros([elem_size,size(dM)]);
epsP_yy = zeros([elem_size,size(dM)]);
epsP_zz = zeros([elem_size,size(dM)]);
epsP_xy = zeros([elem_size,size(dM)]);
%accumulated stresses
sigma_xx = zeros([elem_size,size(dM)]);
sigma_yy = zeros([elem_size,size(dM)]);
sigma_zz = zeros([elem_size,size(dM)]);
sigma_xy = zeros([elem_size,size(dM)]);
%nodal forces
Fx = zeros([elem_size,size(dM)]);
Fy = zeros([elem_size,size(dM)]);
%material parameters in GLL nodes
Vp = zeros([elem_size,size(dM)]);
Vs = zeros([elem_size,size(dM)]);
rho = zeros([elem_size,size(dM)]);
coh = zeros([elem_size,size(dM)]);
phi = zeros([elem_size,size(dM)]);
psi = zeros([elem_size,size(dM)]);
incl = find(SEM_Material > 1);
Vp(:,:,:) = Vp_d;
Vs(:,:,:) = Vs_d;
rho(:,:,:) = rho_d;
coh(:,:,:) = coh_d;
phi(:,:,:) = phi_d;
psi(:,:,:) = psi_d;
%compute Lame parameters in SEM nodes
mu = Vs.*Vs.*rho;
lambda = Vp.*Vp.*rho - 2*mu;
A = 6*coh.*cos(phi)./(3-sin(phi))/sqrt(3);
B = -2*sin(phi)./(3-sin(phi))/sqrt(3);
C = -2*sin(psi)./(3-sin(psi))/sqrt(3);

XYV_elem(1:4) = {zeros(4, N*N*elem_size)}; %list of output elements' coordinates and data field on them
iel = 1:elem_size;

%compute Jacobi matrix for the transformation from the reference to physical coordinates
%X = SUM(Pt_k_l * N_k(ksi) *N_l(eta))
for jm=1:N+1
    Elem = squeeze(SEM_Elements(:,jm,iel));
    dXdKsi(iel,:,jm) = (dM*X(Elem))';
    dYdKsi(iel,:,jm) = (dM*Y(Elem))';
    
    Elem = squeeze(SEM_Elements(jm,:,iel));
    dXdEta(iel,jm,:) =  X(Elem')*(dM.');
    dYdEta(iel,jm,:) =  Y(Elem')*(dM.');
end
Jacobian = dXdKsi.*dYdEta - dYdKsi.*dXdEta;
%GLL quadrature in physical coordinates
dS = abs(Jacobian).*permute(repmat((wL.')*wL,[1,1,elem_size]),[3 1 2]);
Elements = permute(SEM_Elements(:,:,iel),[3 1 2]);

%compute inverse Jacobi matrix
dKsidX =  dYdEta./Jacobian;
dKsidY = -dXdEta./Jacobian;
dEtadX = -dYdKsi./Jacobian;
dEtadY =  dXdKsi./Jacobian;

dist = (xL(2)-xL(1)) / (1/(dXdKsi.*dXdKsi + dYdKsi.*dYdKsi) + 1/(dXdEta.*dXdEta + dYdEta.*dYdEta)).^(1/2);
dt = 0.4*min(dist(:))/Vp_d; %CFL condition for time integration
T_iter = floor(max_Time / dt); %number of transient time iterations

%compute mass matrix's diagonal components
mass_matr = (1+damp*dt)*rho.*dS; %local mass matrices in SEM elements
mass_matr = accumarray(Elements(:),mass_matr(:),[node_size 1])'; %assembly over all SEM elements

f3 = figure(1);clf,colormap jet;
%%%%%%%%%%%%%%%%%%%%SOLVING%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%profile on
tic
cur_step = max_step;
cur_def = 0;
t = 0;
while cur_def < max_Def
    %%%%%%%%%%%%%%%%%%%%%%%%%%SAVE DATA to FILE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fid = fopen('struct_in.dat','wb');
    fwrite(fid,node_size,'int');
    fwrite(fid,elem_size,'int');
    fwrite(fid,T_iter,'int');
    fwrite(fid,xL,'double');
    fwrite(fid,wL,'double');
    fwrite(fid,permute(dM, [2 1]),'double');
    fwrite(fid,dt,'double');
    fwrite(fid,cur_step,'double');
    fwrite(fid,r_e,'double');
    fwrite(fid,damp,'double');
    fwrite(fid,SEM_Nodes,'double');
    fwrite(fid,permute(SEM_Elements, [2 1 3]),'int');
    fwrite(fid,un,'double');

    epsP(:,:,:,1) = epsP_xx;
    epsP(:,:,:,2) = epsP_yy;
    epsP(:,:,:,3) = epsP_zz;
    epsP(:,:,:,4) = epsP_xy;
    fwrite(fid,permute(epsP, [4 3 2 1]),'double');

    fwrite(fid,permute(rho, [3 2 1]),'double');
    fwrite(fid,permute(lambda, [3 2 1]),'double');
    fwrite(fid,permute(mu, [3 2 1]),'double');
    fwrite(fid,permute(A, [3 2 1]),'double');
    fwrite(fid,permute(B, [3 2 1]),'double');
    fwrite(fid,permute(C, [3 2 1]),'double');
    fclose(fid);

    % Action on GPU
    system(['nvcc -gencode=arch=compute_70,code=\"sm_70,compute_70\" -l cublas SEM2D_plastic.cu']);
    if ispc
        tic,system('a.exe');GPU_time = toc;
    else
        tic,system('./a.out');GPU_time = toc;
    end
    
    fid = fopen('struct_out.dat');
    GPU_iter = fread(fid, 1, 'int');
    GPU_dun = fread(fid, [2, node_size], 'double');
    
    GPU_stress = fread(fid, [4*(N+1)*(N+1), elem_size], 'double');
    GPU_stress = reshape(GPU_stress, [4, (N+1), (N+1), elem_size]);
    GPU_stress = permute(GPU_stress, [4 3 2 1]);
    GPU_sigma_xx = GPU_stress(:,:,:,1);
    GPU_sigma_yy = GPU_stress(:,:,:,2);
    GPU_sigma_zz = GPU_stress(:,:,:,3);
    GPU_sigma_xy = GPU_stress(:,:,:,4);
    
    GPU_epsP = fread(fid, [4*(N+1)*(N+1), elem_size], 'double');
    GPU_epsP = reshape(GPU_epsP, [4, (N+1), (N+1), elem_size]);
    GPU_epsP = permute(GPU_epsP, [4 3 2 1]);
    GPU_epsP_xx = GPU_epsP(:,:,:,1);
    GPU_epsP_yy = GPU_epsP(:,:,:,2);
    GPU_epsP_zz = GPU_epsP(:,:,:,3);
    GPU_epsP_xy = GPU_epsP(:,:,:,4);

    fclose(fid);

    cur_def = cur_def + cur_step; %current load

    % Action on CPU
    if matlab_run == 1, tic
        %zero differential displacements and velocities before each load step
        dun = dun*0;
        dvn = dvn*0;
        for iter = 1:T_iter
            UX = dun(1,:)+un(1,:);
            UY = dun(2,:)+un(2,:);
            %compute displacement derivatives in reference and physical coordinates
            for jm=1:N+1
                Elem = squeeze(SEM_Elements(:,jm,iel));
                dUxdKsi(iel,:,jm) = (dM*UX(Elem))';
                dUydKsi(iel,:,jm) = (dM*UY(Elem))';

                Elem = squeeze(SEM_Elements(jm,:,iel));
                dUxdEta(iel,jm,:) =  UX(Elem')*(dM.');
                dUydEta(iel,jm,:) =  UY(Elem')*(dM.');
            end
            dUxdX = dUxdKsi.*dKsidX + dUxdEta.*dEtadX;
            dUxdY = dUxdKsi.*dKsidY + dUxdEta.*dEtadY;
            dUydX = dUydKsi.*dKsidX + dUydEta.*dEtadX;
            dUydY = dUydKsi.*dKsidY + dUydEta.*dEtadY;

            %compute strains for the linear 2D plane strain case
            eps_xx = dUxdX - epsP_xx;
            eps_yy = dUydY - epsP_yy;
            eps_zz = -epsP_zz;
            eps_xy = (dUxdY + dUydX)/2 - epsP_xy;
            EpsV = eps_xx+eps_yy+eps_zz;

            %compute full stresses
            sigma_xx = lambda.*EpsV + 2*mu.*eps_xx;
            sigma_yy = lambda.*EpsV + 2*mu.*eps_yy;
            sigma_zz = lambda.*EpsV + 2*mu.*eps_zz;
            sigma_xy = 2*mu.*eps_xy;

            epsP_xx_new = epsP_xx;
            epsP_yy_new = epsP_yy;
            epsP_zz_new = epsP_zz;
            epsP_xy_new = epsP_xy;

            %check yield criterion
            P = sigma_xx+sigma_yy+sigma_zz;
            F_pl = DruckerPrager(A, B, sigma_xx, sigma_yy, sigma_zz, sigma_xy);
            ipl = find(F_pl > 0 & A+B.*P > 0);
            if ~isempty(ipl) %yield criterion
                bulk = 3*lambda(ipl) + 2*mu(ipl);
                dl = F_pl(ipl)./(mu(ipl) + 3*B(ipl).*C(ipl).*bulk); %plastic multiplier
                %compute stresses using plastic multiplier
                P_pl = P(ipl) + 3*bulk.*C(ipl).*dl;
                S = A(ipl) + B(ipl).*P_pl;
                sigma_xx(ipl) = ((sigma_xx(ipl)+bulk.*C(ipl).*dl).*S+mu(ipl).*dl.*P_pl/3)./(S+mu(ipl).*dl);
                sigma_yy(ipl) = ((sigma_yy(ipl)+bulk.*C(ipl).*dl).*S+mu(ipl).*dl.*P_pl/3)./(S+mu(ipl).*dl);
                sigma_zz(ipl) = ((sigma_zz(ipl)+bulk.*C(ipl).*dl).*S+mu(ipl).*dl.*P_pl/3)./(S+mu(ipl).*dl);
                sigma_xy(ipl) = sigma_xy(ipl).*S./(S+mu(ipl).*dl);
                %plastic strains
                epsP_xx_new(ipl) = epsP_xx_new(ipl) + dl.*((sigma_xx(ipl)-P_pl/3)./S/2-C(ipl));
                epsP_yy_new(ipl) = epsP_yy_new(ipl) + dl.*((sigma_yy(ipl)-P_pl/3)./S/2-C(ipl));
                epsP_zz_new(ipl) = epsP_zz_new(ipl) + dl.*((sigma_zz(ipl)-P_pl/3)./S/2-C(ipl));
                epsP_xy_new(ipl) = epsP_xy_new(ipl) + dl.*sigma_xy(ipl)./S/2;
            end

            %integrate stresses multiplied by derivatives of shape functions
            sigmax_ksi = (sigma_xx.*dKsidX + sigma_xy.*dKsidY).*dS;
            sigmax_eta = (sigma_xx.*dEtadX + sigma_xy.*dEtadY).*dS;
            sigmay_ksi = (sigma_xy.*dKsidX + sigma_yy.*dKsidY).*dS;
            sigmay_eta = (sigma_xy.*dEtadX + sigma_yy.*dEtadY).*dS;

            %compute nodal forces (F = K*U, K-stiffness matrix)
            for jm=1:N+1
                Fx(iel,:,jm) = ((dM.')*sigmax_ksi(iel,:,jm)')';
                Fy(iel,:,jm) = ((dM.')*sigmay_ksi(iel,:,jm)')';
            end
            for im=1:N+1
                Fx(iel,im,:) = squeeze(Fx(iel,im,:)) + (squeeze(sigmax_eta(iel,im,:))*dM);
                Fy(iel,im,:) = squeeze(Fy(iel,im,:)) + (squeeze(sigmay_eta(iel,im,:))*dM);
            end

            %damping
            VX = dvn(1,:);
            VY = dvn(2,:);
            Fx = Fx + damp*VX(Elements).*rho.*dS;
            Fy = Fy + damp*VY(Elements).*rho.*dS;

            %compute accelerations by assemblying nodal forces over all SEM mesh
            dan(1, :) = (-accumarray(Elements(:),Fx(:),[node_size 1])') ./ mass_matr;
            dan(2, :) = (-accumarray(Elements(:),Fy(:),[node_size 1])') ./ mass_matr;

            %Newmark scheme 2nd order in time
            dvn = dvn + dan*dt;
            dun = dun + dvn*dt + dan*dt*dt/2;

            %set Dirichlet's BC
            eps = min(Lx,Ly)/1.e7;
            %at external circular boundary
            ind = find(X.^2+Y.^2 > r_e^2-eps);
            dun(1,ind) = cur_step*X(ind);
            dun(2,ind) = -cur_step*Y(ind);
            dvn(1,ind) = 0;
            dvn(2,ind) = 0;
            %at external bottom boundary
            ind = find(Y < eps);
            dun(2,ind) = 0;
            dvn(2,ind) = 0;
            %at external rectangular's edges
            %dun(1,abs(X-Lx/2) < eps) = -cur_step*Lx;
            %dun(1,abs(X+Lx/2) < eps) = cur_step*Lx;
            %dun(2,abs(Y) < eps) = cur_step*Ly;
            %dun(2,abs(Y+Ly) < eps) = -cur_step*Ly;
            %dvn(1,abs(X-Lx/2) < eps) = 0;
            %dvn(1,abs(X+Lx/2) < eps) = 0;
            %dvn(2,abs(Y) < eps) = 0;
            %dvn(2,abs(Y+Ly) < eps) = 0;

            %check convergence of nonlinear iterations
            if norm(dvn(:)) < 1.e-5 && mod(iter, 1000) == 0
                break;
            end
        end
        CPU_time   = toc;
        speedup    = CPU_time/GPU_time

        diff_sigma_xx = sigma_xx - GPU_sigma_xx;
        diff_sigma_yy = sigma_yy - GPU_sigma_yy;
        diff_sigma_zz = sigma_zz - GPU_sigma_zz;
        diff_sigma_xy = sigma_xy - GPU_sigma_xy;

        Mdiff_sigma_xx = max(abs(diff_sigma_xx(:))) / max(abs(sigma_xx(:))) * 100
        Mdiff_sigma_yy = max(abs(diff_sigma_yy(:))) / max(abs(sigma_yy(:))) * 100
        Mdiff_sigma_zz = max(abs(diff_sigma_zz(:))) / max(abs(sigma_zz(:))) * 100
        Mdiff_sigma_xy = max(abs(diff_sigma_xy(:))) / max(abs(sigma_xy(:))) * 100
    else
        iter = GPU_iter;
    end

    %if disconverged => revert back to previous step and half the load step
    if iter >= T_iter && cur_step > max_step*dt/10
        cur_def = cur_def - cur_step;
        cur_step = cur_step / 2;
        continue;
    else
        cur_step = min(cur_step*1.1, max_step);
    end

    if matlab_run ~= 1
        dun = GPU_dun;

        sigma_xx = GPU_sigma_xx;
        sigma_yy = GPU_sigma_yy;
        sigma_zz = GPU_sigma_zz;
        sigma_xy = GPU_sigma_xy;

        epsP_xx_new = GPU_epsP_xx;
        epsP_yy_new = GPU_epsP_yy;
        epsP_zz_new = GPU_epsP_zz;
        epsP_xy_new = GPU_epsP_xy;
    end

    %update full displacements
    un = un + dun;
    %update accumulated plastic strains
    epsP_xx = epsP_xx_new;
    epsP_yy = epsP_yy_new;
    epsP_zz = epsP_zz_new;
    epsP_xy = epsP_xy_new;

    P = sigma_xx+sigma_yy+sigma_zz;
    UX = un(1,:);
    UY = un(2,:);

    %%%%%%%%%%%%%%%%%%%%%%%%%POSTPROCESSING%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t = t+1; %physical time step
    if mod(t,T_vis)==0 || t < 100
        %compute plastic strain intensity
        epsP_i = (((epsP_xx-epsP_yy).^2+(epsP_yy-epsP_zz).^2+(epsP_zz-epsP_xx).^2)/6+epsP_xy.^2).^(1/2);
        P = sigma_xx+sigma_yy+sigma_zz;
        
        for i = 1:elem_size %fill output arrays
            Elem = SEM_Elements(:,:,i);
            %OUTPUT = {X(Elem), Y(Elem), VX(Elem).*VX(Elem)+VY(Elem).*VY(Elem)};
            OUTPUT = {X(Elem)+UX(Elem), Y(Elem)+UY(Elem), squeeze(-P(i,:,:)), squeeze(epsP_i(i,:,:))};
            for k = 1:N %split SEM into quadrangles
                for l = 1:N
                    cur_ind = l + (k-1)*N + (i-1)*N*N;
                    for m = 1:4
                        XYV_elem{m}(1, cur_ind) = OUTPUT{m}(k,l);
                        XYV_elem{m}(2, cur_ind) = OUTPUT{m}(k+1,l);
                        XYV_elem{m}(3, cur_ind) = OUTPUT{m}(k+1,l+1);
                        XYV_elem{m}(4, cur_ind) = OUTPUT{m}(k,l+1);
                    end
                end
            end
        end
        
        figure(f3);
        subplot(2,2,1);
        newplot;
        patch(XYV_elem{1}, XYV_elem{2}, XYV_elem{3}, 'EdgeColor','none'),title("Pressure"),colorbar,axis image;
        
        subplot(2,2,2);
        newplot;
        patch(XYV_elem{1}, XYV_elem{2}, XYV_elem{4}, 'EdgeColor','none'),title("Plastic strain intensity"),colorbar,axis image;
        
        subplot(2,2,[3,4]);
        %semilogy(t*dt,max(abs(sigma_yy(:)/bc_stress)),'*'),axis tight,title('max abs Stress'),hold on;
        %semilogy(t*dt,max(abs(vn(1,:))+abs(vn(2,:))),'*'),axis tight,title('max abs Velocity'),hold on;
        plot(cur_def,iter,'*'),axis tight,title('Number of nonlinear iterations'),hold on;
        drawnow;
        
        [Pic,map] = rgb2ind(frame2im(getframe(f3)),256);
        if t == 1
            imwrite(Pic,map,'plastic.gif','gif','LoopCount',Inf,'DelayTime',0.1);
        else
            imwrite(Pic,map,'plastic.gif','gif','WriteMode','append','DelayTime',0.1);
        end
    end
end
toc
%profile viewer

function [Vp, Vs] = Velocity(K, G, rho)
Vp = sqrt((K+4*G/3)/rho); %longitudinal wave speed
Vs = sqrt(G/rho); %shear wave speed
end

%compute Cauchy stress tensor
function [sigma_xx, sigma_yy, sigma_zz, sigma_xy] = Cauchy(lambda, mu, J, Bxx, Byy, Bzz, Bxy)
%Chernykh material
I = (lambda.*(Bxx+Byy+Bzz-3)/2-mu)./J;
sigma_xx = I.*Bxx + mu.*(Bxx.*Bxx+Bxy.*Bxy);
sigma_yy = I.*Byy + mu.*(Bxy.*Bxy+Byy.*Byy);
sigma_zz = I.*Bzz + mu.*Bzz.*Bzz;
sigma_xy = I.*Bxy + mu.*(Bxx.*Bxy+Bxy.*Byy);
end

%Drucker-Prager plastic criterion
function F = DruckerPrager(A, B, sigma_xx, sigma_yy, sigma_zz, sigma_xy)
P = sigma_xx+sigma_yy+sigma_zz;
F = (((sigma_xx-P/3).^2+(sigma_yy-P/3).^2+(sigma_zz-P/3).^2)/2+sigma_xy.^2).^(1/2) - A - B.*P;
end