import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import grid

import torch
import torchkbnufft as tkbn


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# load matlab file
mat = scipy.io.loadmat('radial_data.mat')
kspace_radial = mat['k']  # 384×600

def ifft2c(kspace):
    kspace = np.fft.ifftshift(kspace)
    image = np.fft.ifft2(kspace)
    image = np.fft.fftshift(image)
    return image

# plt.figure()
# plt.imshow(np.log(np.abs(kspace)), cmap='gray')
# plt.title('kspace with logirithm')
# plt.show()
# plt.savefig('./result_image/kspace with logirithm.png')


#radial sampling advantage：get a lot of signal in each repetition;
# not sensitive to motion artifact; fast
#######################################################
###### 1.Radial sampling pattern ######
#######################################################
def radial_trajectory(delta_ang, start_ang, kspace):
    lenth, num_lin = np.shape(kspace) #384,600
    sample_points = np.zeros_like(kspace,dtype=complex) # 注意dtype，否则下方的语句执行后Complex_Mat中只存放实数

    for j in range(num_lin):  # 600
        angle = np.deg2rad(j * delta_ang + start_ang)

        for l in range(lenth):  # 384
            x = (l - (lenth-1)/2) * np.cos(angle)/lenth # scale from -0.5 to 0.5
            y = (l - (lenth-1)/2) * np.sin(angle)/lenth
            sample_points[l, j] = x+1j*y

        # plt.plot([np.real(sample_points[0, j]), np.real(sample_points[-1, j])],
        #          [np.imag(sample_points[0, j]), np.imag(sample_points[-1, j])])

    return sample_points

golden_angle_increment = 111.246117975
trajectory = radial_trajectory(golden_angle_increment, 90, kspace_radial)

# plt.title('kspace trajectory: 600 spokes')
# plt.show()


#######################################################
###### 2. Basic gridding reconstruction ######
#######################################################
kspace_catesian_grid = grid.grid(kspace_radial, trajectory,np.shape(kspace_radial)[0])  #384×384
img_catesian_grid = ifft2c(kspace_catesian_grid)

# plt.subplot(121)
# plt.imshow(np.log(np.abs(kspace_catesian_grid)),cmap='gray')
# plt.title('catesian_grid in kspace')
# plt.subplot(122)
# plt.imshow(np.abs(img_catesian_grid),cmap='gray')
# plt.title('reconstructed MRI img of catesian_grid')
# plt.savefig('./result_image/catesian_grid.png')
# plt.show()


#######################################################
###### 3. Density compensation######
#######################################################
def density_compensation(kspace):
    kp_sz = len(kspace)
    ramp1D_half = np.linspace(1,1/kp_sz,kp_sz//2)
    ramp1D = np.concatenate((ramp1D_half, ramp1D_half[::-1]))
    ramp2D = np.sqrt(np.outer(ramp1D, ramp1D))
    kspace_denComp = kspace * ramp2D
    img_denComp = ifft2c(kspace_denComp)
    return ramp2D, kspace_denComp, img_denComp

ramp2D, kspace_denComp, img_denComp= density_compensation(kspace_catesian_grid)


# plt.subplot(131)
# plt.imshow(np.abs(ramp2D),cmap='gray')
# plt.title('ramp filter')
# plt.subplot(132)
# plt.imshow(np.log(np.abs(kspace_denComp)),cmap='gray')
# plt.title('kspace after density compensation')
# plt.subplot(133)
# plt.imshow(np.abs(img_denComp),cmap='gray')
# plt.title('reconstructed MRI img')
# plt.savefig('./result_image/density compensation.png')
# plt.show()


#######################################################
###### 4. Oversampling ######
#######################################################
'''''
kspace_overSamHalf = grid.grid(kspace_radial, trajectory, int(np.shape(kspace_radial)[0]*1.5))
_,kspace_overSamHalf,_ = density_compensation(kspace_overSamHalf)
img_overSamHalf = ifft2c(kspace_overSamHalf)
sz_overSamHalf = len(kspace_overSamHalf)
sz_crop = len(kspace_radial)
img_overSamHalf_crop = img_overSamHalf[sz_overSamHalf//2-sz_crop//2: sz_overSamHalf//2+sz_crop//2,
                                       sz_overSamHalf//2-sz_crop//2: sz_overSamHalf//2+sz_crop//2]
'''''

# plt.subplot(131)
# plt.imshow(np.angle(kspace_overSamHalf),cmap='gray')
# plt.title('1.5factor oversampling kspace')
# plt.subplot(132)# plt.imshow(np.abs(img_pad_crop2),cmap='gray')
# plt.imshow(np.abs(img_overSamHalf),cmap='gray')
# plt.title('1.5factor oversamp img')
# plt.subplot(133)
# plt.imshow(np.abs(img_overSamHalf_crop),cmap='gray')
# plt.title('croped img')
# plt.show()

sz_crop = len(kspace_radial)
kspace_overSam2 = grid.grid(kspace_radial, trajectory, int(np.shape(kspace_radial)[0]*2))
_,kspace_overSam2,_ = density_compensation(kspace_overSam2)
img_overSam2 = ifft2c(kspace_overSam2)
sz_overSam2 = len(kspace_overSam2)
img_overSam2_crop = img_overSam2[sz_overSam2//2-sz_crop//2: sz_overSam2//2+sz_crop//2,
                                    sz_overSam2//2-sz_crop//2: sz_overSam2//2+sz_crop//2]

# plt.subplot(131)
# plt.imshow(np.angle(kspace_overSam2),cmap='gray')
# plt.title('2factor oversampling kspace')
# plt.subplot(132)
# plt.imshow(np.abs(img_overSam2),cmap='gray')
# plt.title('2factor oversamp img')
# plt.subplot(133)
# plt.imshow(np.abs(img_overSam2_crop),cmap='gray')
# plt.title('croped img')
# plt.show()

'''''
Comparison between oversampling and zero padding
Oversampling通过增加kspac sample points(△k变小)增加Fov. Fov的意思是增加视野，(Fov ~ 1/△k)
Zeropadding 不会增加Fov,(看到的东西没有变多)
'''''
zero_pad = np.pad(kspace_denComp,  ((192, 192), (192, 192)), 'constant', constant_values=(0, 0))
img = ifft2c(zero_pad)

# plt.subplot(221)
# plt.imshow(np.angle(zero_pad),cmap='gray')
# plt.title('zero_pad kspace')
# plt.subplot(222)
# plt.imshow(np.abs(img),cmap='gray')
# plt.title('zero_pad img')
# plt.subplot(223)
# plt.imshow(np.angle(kspace_overSam2),cmap='gray')
# plt.title('2factor oversampling kspace')
# plt.subplot(224)
# plt.imshow(np.abs(img_overSam2),cmap='gray')
# plt.title('2factor oversamp img')
# plt.show()


#######################################################
###### 5.De-apodization ######
#######################################################
tri1D = [0, 0.5, 1, 0.5, 0]
tri2D = np.sqrt(np.outer(tri1D, tri1D))
pad_w = (sz_overSam2-len(tri1D))//2
tri2D_pad = np.pad(tri2D, ((pad_w+1,pad_w), (pad_w+1,pad_w)), 'constant', constant_values=0)
tri_ift = ifft2c(tri2D_pad)+10**(-5)
img_deApo = img_overSam2/tri_ift
img_deApo_crop = img_deApo[(sz_overSam2-sz_crop)//2:(sz_overSam2+sz_crop)//2,
                 (sz_overSam2-sz_crop)//2:(sz_overSam2+sz_crop)//2]

plt.subplot(131)
plt.imshow(np.abs(tri_ift),cmap='gray')
plt.title('ifft of kernel')
plt.subplot(132)
plt.imshow(np.abs(img_overSam2_crop),cmap='gray')
plt.title('Without de-apodization,crop')
plt.subplot(133)
plt.imshow(np.abs(img_deApo_crop),cmap='gray')
plt.title('With de-apodization,crop')
plt.show()

# plt.subplot(221)
# plt.plot(np.abs(img_overSam2_crop[192,:]))
# plt.title('Without de-apodization')
# plt.subplot(222)
# plt.plot(np.abs(img_deApo_crop[192,:]))
# plt.title('With de-apodization')
# plt.subplot(223)
# plt.imshow(np.abs(img_overSam2_crop),cmap='gray')
# plt.title('Without de-apodization')
# plt.axis('off')
# plt.subplot(224)
# plt.imshow(np.abs(img_deApo_crop),cmap='gray')
# plt.title('With de-apodization')
# plt.axis('off')
# plt.show()


#######################################################
###### 6. NUFFT toolbox ######
#######################################################
'''''
@conference{muckley:20:tah,
  author = {M. J. Muckley and R. Stern and T. Murrell and F. Knoll},
  title = {{TorchKbNufft}: A High-Level, Hardware-Agnostic Non-Uniform Fast {Fourier} Transform},
  booktitle = {ISMRM Workshop on Data Sampling \& Image Reconstruction},
  year = 2020,
  note = {Source code available at https://github.com/mmuckley/torchkbnufft}.
}

'''

spokelength, nspokes = np.shape(kspace_radial)
ga = np.deg2rad(golden_angle_increment)
kx = np.zeros((spokelength, nspokes))
ky = np.zeros((spokelength, nspokes))
ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
for i in range(1, nspokes):
    kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
    ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]
ky = np.transpose(ky)
kx = np.transpose(kx)

ktraj = np.stack((ky.flatten(), kx.flatten()), axis=0)
ktraj = torch.tensor(ktraj)

kdata = kspace_radial.transpose()
kdata = kdata.reshape((1,-1))
kdata = torch.tensor(kdata).unsqueeze(0)

adjnufft_ob = tkbn.KbNufftAdjoint(im_size=(sz_crop,sz_crop))

image = adjnufft_ob(kdata, ktraj)
image_blurry_numpy = np.squeeze(image.numpy())
image_blurry_numpy = image_blurry_numpy.transpose()

dcomp = tkbn.calc_density_compensation_function(ktraj=ktraj, im_size=(sz_crop,sz_crop))
image_sharp = adjnufft_ob(kdata * dcomp, ktraj)
image_sharp_numpy = np.squeeze(image_sharp.numpy())
image_sharp_numpy = image_sharp_numpy.transpose()


# plt.subplot(221)
# plt.imshow(np.abs(img_catesian_grid),cmap='gray')
# plt.title('Nan: blurry img')
# plt.subplot(222)
# plt.imshow(np.abs(img_denComp),cmap='gray')
# plt.title('Nan: sharp img')
# plt.subplot(223)
# plt.imshow(np.abs(image_blurry_numpy),cmap='gray')
# plt.title('NUFFT toolbox:blurry img')
# plt.subplot(224)
# plt.imshow(np.abs(image_sharp_numpy),cmap='gray')
# plt.title('NUFFT toolbox:sharp img')
# plt.show()

