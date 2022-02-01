function [ x ] = fft2_mri( y )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
[kx,ky] = size(y);
x = fftshift(fft2(ifftshift(y/sqrt(kx*ky))));
end

