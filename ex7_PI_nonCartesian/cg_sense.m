function  u = cg_sense(data,FT,c,mask,alpha,tol,maxit,matlabCG,display)
% 
% [u] = cg_sense(data,FT,c,mask,alpha,maxit,display)
% reconstruct subsampled PMRI data using CG SENSE [1]
%
% INPUT
% data:    3D array of coil images (in k-space)
% FT:      NUFFT operator
% c:       3D array of coil sensitivities
% mask:    region of support for sampling trajectory
% alpha:   penalty for Tikhonov regularization
% maxit:   maximum number of CG iterations
% display: show iteration steps (1) or not (0)
% 
% OUTPUT
% u:       reconstructed image
%
% Last Change: 29.05.2012
% By: Florian (florian.knoll@tugraz.at)
% 
% [1] Pruessmann, K. P.; Weiger, M.; Boernert, P. and Boesiger, P.
% Advances in sensitivity encoding with arbitrary k-space trajectories.
% Magn Reson Med 46: 638-651 (2001)
% 
% =========================================================================

%% set up parameters and operators
[nx,ny] = size(FT'*data(:,:,1));
nc      = size(c,3);

%% sampling operator
F  = @(x) FT*x;
FH = @(x) FT'*x;

%% Solve using CG method
% precompute complex conjugates
cbar = conj(c);

% right hand side: -K^*residual 
y  = zeros(nx,ny);
for i = 1:nc
    y = y + FH(data(:,:,i)).*cbar(:,:,i);
end

% system matrix: F'^T*F' + alpha I
M  = @(x) applyM(F,FH,c,cbar,x) + alpha*x;

%% CG iterations
if matlabCG
    % Use Matlab CG
    x = pcg(M,y(:),tol,maxit);
else
    % Own CG
    x = 0*y(:); r=y(:); p = r; rr = r'*r; it=0;
    while(rr>tol*norm(r))&&(it<maxit)
        Ap = M(p);
        a = rr/(p'*Ap);
        x = x + a*p;
        rnew = r - a*Ap;
        b = (rnew'*rnew)/rr;
        r=rnew;
        rr = r'*r;
        p = r + b*p;
        it=it+1;
        
        if display
            u_it = reshape(x,nx,ny);
            kspace_it = fft2c(u_it).*mask;
            u_it =ifft2c(kspace_it);
            figure(99);
            subplot(1,2,1),imshow(abs(u_it),[]); % colorbar;
            title(['Image CG iteration ' num2str(it)]);
            subplot(1,2,2),kshow(kspace_it);
            title(['k-space iteration ' num2str(it)]);
            drawnow;
        end
    end
    disp(['Own CG: iter ', num2str(it), ' tol: ', num2str(rr/norm(r))]);
end

% Final reconstructed image
u  = reshape(x,nx,ny);

% Mask k-space with region of support of trajectory
u =ifft2c(fft2c(u).*mask);

% end main function

%% Derivative evaluation
function y = applyM(F,FH,c,cconj,x)
[nx,ny,nc] = size(c);
dx = reshape(x,nx,ny);

y  = zeros(nx,ny);
for i = 1:nc
    y = y + cconj(:,:,i).*FH(F(c(:,:,i).*dx));
end
y = y(:);
% end function applyM
