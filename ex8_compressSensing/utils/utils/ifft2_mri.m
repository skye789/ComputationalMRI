function [ y ] = ifft2_mri( x )
[kx,ky] = size(x);
y = fftshift(ifft2(ifftshift(x*sqrt(kx*ky))));
end

