addpath(genpath('Wavelab850'));

load data_lab6;

fullim = ifft2_mri(kfull);
wave_coeff = fdwt_db4(fullim);
figure(1);
imagesc(abs(wave_coeff), [0 1]); colormap gray;

thresh = prctile(abs(wave_coeff), 80, 'all');

wave_coeff_down = wave_coeff;
wave_coeff_down(abs(wave_coeff) < thresh) = 0;

down_im = idwt_db4(wave_coeff_down);

figure(2);
imagesc(abs(down_im), [0 1]); colormap gray;