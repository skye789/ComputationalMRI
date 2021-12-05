import matplotlib.pyplot
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import grid
import sys


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
triangle = [0, 0.5, 1,0.5,0]
tri = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(triangle)))

# img_tri = np.zeros_like(img_overSam2_crop)
# for i in range(len(kspace_radial)-4):
#     img_tri = np.abs(img_overSam2_crop[:, i:i+5])/np.abs(tri)
# for i in range(len(kspace_radial-4)):
#     img_tri = img_overSam2_crop[i:i+5,:]/tri

# img_deApo = img_overSam2_crop/triangle

a = np.abs(img_tri)-np.abs(img_overSam2_crop)
print(np.nonzero(a))
plt.subplot(221)
plt.imshow(np.abs(img_tri), cmap='gray')
plt.subplot(222)
plt.imshow(np.abs(img_tri)-np.abs(img_overSam2_crop), cmap='gray')
plt.show()
