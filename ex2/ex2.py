import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hamming

# load matlab file
mat = scipy.io.loadmat('kdata1.mat')
kspace = mat['kdata1']


###### 1. FFT reconstruction of Cartesian MRI data
def fft2c(img):
    kspace = np.fft.fft2(img)
    kspace = np.fft.fftshift(kspace)
    return kspace


def ifft2c(kspace):
    image = np.fft.ifft2(kspace)
    img = np.fft.ifftshift(image)
    return img


recon_image = ifft2c(kspace)
magnitube = np.abs(recon_image)
angle = np.angle(recon_image, deg=True)

# plt.figure()
# plt.subplot(121)
# plt.imshow(magnitube, cmap='gray')
# plt.title('magnitude of recon_image')
# plt.subplot(122)
# plt.imshow(angle, cmap='gray')
# plt.title('phase of recon_image')
# plt.show()


###### 2. Effects of k-space zero-padding and zero padding
def get_truncated_kspace(kspace,width):
    half_w = width//2
    kspace = kspace[256 - half_w:256 + half_w, 256 - half_w:256 + half_w]
    return kspace

def zero_padding(kspace):
    pad_num = 256-len(kspace)//2
    zero_pad_kspace = np.pad(kspace, ((pad_num, pad_num), (pad_num, pad_num)), 'constant', constant_values=0)
    return zero_pad_kspace

kspace_64 = get_truncated_kspace(kspace, 64)
kspace_64_pad = zero_padding(kspace_64)
kspace_128 = get_truncated_kspace(kspace, 64)
kspace_128_pad = zero_padding(kspace_128)
kspace_256 = get_truncated_kspace(kspace, 64)
kspace_256_pad = zero_padding(kspace_256)

recon_image_64 = ifft2c(kspace_64_pad)
recon_image_128 = ifft2c(kspace_128_pad)
recon_image_256 = ifft2c(kspace_256_pad)

# plt.figure()
# plt.subplot(221)
# plt.imshow(np.abs(recon_image_64), cmap='gray')
# plt.title('recon_image from 64×64 kspace')
# plt.axis('off')
# plt.subplot(222)
# plt.imshow(np.abs(recon_image_128), cmap='gray')
# plt.title('recon_image from 128×128 kspace')
# plt.axis('off')
# plt.subplot(223)
# plt.imshow(np.abs(recon_image_256), cmap='gray')
# plt.title('recon_image from 256×256 kspace')
# plt.axis('off')
# plt.subplot(224)
# plt.imshow(np.abs(recon_image), cmap='gray')
# plt.title('recon_image from original kspace')
# plt.axis('off')
# plt.show()


###### 3. Point spread function (PSF)
# kspace · psf;    PSF就是sinc函数，PSF和什么卷积，另一个domain就会模糊
def get_PSF_fre(width):
    cols = len(kspace[0])
    psf = np.zeros((cols))
    psf[cols // 2 -width//2 : cols // 2 + width//2] = 1
    psf_fre = np.fft.fftshift(np.fft.fft(psf))
    return psf_fre

def get_FWHM(psf_fre):
    half_maximum = max(np.abs(psf_fre))//2
    new_fun = np.abs(psf_fre)-half_maximum
    left = np.argmin(np.abs(new_fun[:len(psf_fre)//2]))
    FWHM = (len(psf_fre)//2-left)*2
    return left, half_maximum, FWHM

psf_fre2= get_PSF_fre(32)
left2,half_maximum2, FWHM2 = get_FWHM(psf_fre2)
psf_fre6 = get_PSF_fre(64)
left6,half_maximum6, FWHM6 = get_FWHM(psf_fre6)
psf_fre10 = get_PSF_fre(128)
left10, half_maximum10, FWHM10 = get_FWHM(psf_fre10)

print('FWHM is respectively', FWHM2,FWHM6,FWHM10)

x = np.linspace(0,511,512)

plt.title('1D Point Spread Function')
plt.plot(x, np.abs(psf_fre2), color='green', label='FWHM 20')
plt.plot(x[left2:left2+FWHM2], np.ones((FWHM2))*half_maximum2, color='green')
plt.plot(x, np.abs(psf_fre6), color='red', label='FWHM 10')
plt.plot(x[left6:left6+FWHM6], np.ones((FWHM6))*half_maximum6, color='red')
plt.plot(x, np.abs(psf_fre10), color='YELLOW', label='FWHM 4')
plt.plot(x[left10:left10+FWHM10], np.ones((FWHM10))*half_maximum10, color='yellow', ls = '--')
plt.legend() # 显示图例
plt.xlabel('frequency')
plt.ylabel('amplitude')
plt.show()



###### 4. k-space filtering (windowing)

window64 = hamming(64)
# window64_time = np.fft.ifftshift(np.fft.ifft(window64))
filter_img = np.zeros_like(kspace)
for i in range(len(kspace)):
    filter_img[i] = np.fft.ifftshift(np.fft.ifft(window64*kspace_64_pad[i]))

# a = scipy.signal.convolve()

# plt.figure()
# plt.subplot(121)
# plt.imshow(np.abs(filter_img), cmap='gray')
# plt.show()



###### 5. Oversampling the readout dimension
mat = scipy.io.loadmat('kdata2.mat')
kspace2 = mat['kdata2'] #168×336
oversampling_img = ifft2c(kspace2)
recon_kspace2 = kspace2[:,0::2] #168×168
normalsampling_img = ifft2c(recon_kspace2)

# plt.figure()
# plt.subplot(121)
# plt.imshow(np.abs(oversampling_img), cmap='gray')
# plt.title('oversampling_img')
# plt.subplot(122)
# plt.imshow(np.abs(normalsampling_img), cmap='gray')
# plt.title('normalsampling_img')
# plt.show()

