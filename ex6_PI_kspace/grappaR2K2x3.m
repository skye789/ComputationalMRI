function krecon=grappaR2K2x3(kdata,acs,flag_acs)
%-------------------------------------------------------------------------
% GRAPPA reconstruction of a 2D image with uniform 2-fold acceleration 
% along the ky dimension using a 2x3 kernel.  
%-------------------------------------------------------------------------
%	Input:
%	kdata: accelerated acquisition [ny/2,nx,nc]. 
%	acs: autocalibration signal  [nyacs,nxacs,nc]. 
%   flag_acs: 1, include the acs data into the final reconstruction
%
%	Output:
%	krecon: coil-by-coil k-space recon [ny,nx,nc].
%--------------------------------------------------------------------------
% Ricardo Otazo, Practical MRI 2 course
%--------------------------------------------------------------------------
[ny,nx,~]=size(kdata);           
[nyacs,nxacs,nc]=size(acs);
% calibration matrix
% initial source matrix
src=zeros((nyacs-2)*(nxacs-2),nc*2*3);
% initial target matrix
targ=zeros((nyacs-2)*(nxacs-2),nc);
src_idx=0;                         
for xi=2:nxacs-1,
for yi=1:nyacs-2,
    src_idx=src_idx+1;
    % collects a 2x3 block of source points around the target point
    block=[];
    for bxi=-1:1,
    for byi=0:1,
        block=cat(1,block,squeeze(acs(yi+byi*2,xi+bxi,:)));
    end
    end
    src(src_idx,:)=block;
    % target point for these source points
    targ(src_idx,:)=squeeze(acs(yi+1,xi,:));                                                          
end
end
% weights using pseudoinverse of the calibration matrix
ws=pinv(src'*src)*src'*targ;      
% apply weights to reconstruct missing k-space points
krecon=zeros(ny*2+1,nx+2,nc);
% known data in the reconstructed k-space matrix
krecon(1:2:end-1,2:end-1,:)=kdata;                         
for xi=2:nx+1,
for yi=1:2:ny*2,
    src=[];
    for bxi=-1:1,
    for byi=0:1,
        src=cat(1,src,squeeze(krecon(yi+byi*2,xi+bxi,:)));
    end
    end
    % recon=source*weights
    krecon(yi+1,xi,:)=transpose(src)*ws;
end
end
% crop the zero-pad at the border
krecon=krecon(1:end-1,2:end-1,:);
% include the calibration data
if flag_acs,
    [ny,nx,~]=size(krecon);
    krecon(ny/2-nyacs/2+1:ny/2+nyacs/2,nx/2-nxacs/2+1:nx/2+nxacs/2,:)=acs;
end