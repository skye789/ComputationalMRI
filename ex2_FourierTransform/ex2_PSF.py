import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# load matlab file
mat = scipy.io.loadmat('kdata1.mat')
kspace = mat['kdata1']


###### 1. FFT reconstruction of Cartesian MRI data
def fft2c(img): #img原点在左上
    kspace = np.fft.fft2(img, axes=(0, 1))
    kspace = np.fft.fftshift(kspace, axes=(0, 1))
    return kspace


def ifft2c(kspace):
    image = np.fft.ifft2(kspace, axes=(0, 1))
    img = np.fft.ifftshift(image, axes=(0, 1))
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
kspace_128 = get_truncated_kspace(kspace, 128)
kspace_128_pad = zero_padding(kspace_128)
kspace_256 = get_truncated_kspace(kspace, 256)
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
#傅里叶变换的卷积要求从负无穷到正无穷积分，原图是512×512，即使kspace也是512×512，
#但是也是truncation(加了一个512×512的rectangular window)
def get_rectan_PSF_fre(width):
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

psf_fre2= get_rectan_PSF_fre(64)
left2,half_maximum2, FWHM2 = get_FWHM(psf_fre2)
psf_fre6 = get_rectan_PSF_fre(128)
left6,half_maximum6, FWHM6 = get_FWHM(psf_fre6)
psf_fre10 = get_rectan_PSF_fre(256)
left10, half_maximum10, FWHM10 = get_FWHM(psf_fre10)

# print('FWHM of rectangular window is respectively', FWHM2,FWHM6,FWHM10)
# x = np.linspace(0,511,512)
# plt.title('1D Point Spread Function of rectangular window')
# plt.plot(x, np.abs(psf_fre2), color='green', label='64×64truncation: FWHM 10')
# plt.plot(x[left2:left2+FWHM2], np.ones((FWHM2))*half_maximum2, color='green')
# plt.plot(x, np.abs(psf_fre6), color='red', label='128×128truncation:FWHM 4')
# plt.plot(x[left6:left6+FWHM6], np.ones((FWHM6))*half_maximum6, color='red')
# plt.plot(x, np.abs(psf_fre10), color='YELLOW', label='256×256truncation:FWHM 2')
# plt.plot(x[left10:left10+FWHM10], np.ones((FWHM10))*half_maximum10, color='yellow', ls = '--')
# plt.legend() # 显示图例
# plt.xlabel('frequency')
# plt.ylabel('amplitude')
# plt.show()


###### 4. k-space filtering (windowing)
##4.1 Image with haming window
def get_hamming_img(kspace, win_sz):
    window_1d = signal.hamming(win_sz)
    window_2d = np.sqrt(np.outer(window_1d, window_1d))
    window_pad_2d = zero_padding(window_2d)
    ham_img = ifft2c(window_pad_2d * kspace)
    return ham_img, window_pad_2d

ham_img_64,_ = get_hamming_img(kspace_64_pad, 64)
ham_img_128,_ = get_hamming_img(kspace_128_pad, 128)
ham_img_256, _ = get_hamming_img(kspace_256_pad, 256)

# plt.figure()
# plt.subplot(131)
# plt.imshow(np.abs(ham_img_64), cmap='gray')
# plt.title('image with 64×64 ham_filter')
# plt.axis('off')
# plt.subplot(132)
# plt.imshow(np.abs(ham_img_128), cmap='gray')
# plt.title('image with 128×128 ham_filter')
# plt.axis('off')
# plt.subplot(133)
# plt.imshow(np.abs(ham_img_256), cmap='gray')
# plt.title('image with 256×256 ham_filter')
# plt.axis('off')
# plt.show()

plt.figure()
plt.subplot(121)
plt.imshow(np.abs(ham_img_128), cmap='gray')
plt.title('image with 128×128 hamming_window')
plt.axis('off')
plt.subplot(122)
plt.imshow(np.abs(recon_image_128), cmap='gray')
plt.title('image with 128×128 rectangular_window')
plt.axis('off')
plt.show()

##4.2 PSF of Hamming window
def get_ham_psf(ham_sz):
    half_sz = ham_sz//2
    window_64 = signal.hamming(ham_sz)
    window_64 = np.pad(window_64, (256 - half_sz, 256 - half_sz), constant_values=0)
    ham_window_64 = np.fft.fftshift(np.fft.fft(window_64))
    return ham_window_64

ham_window_64 = get_ham_psf(64)
ham_window_128 = get_ham_psf(128)
ham_window_256 = get_ham_psf(256)

left64, half_maximum64, FWHM64 = get_FWHM(ham_window_64)
left128, half_maximum128, FWHM128 = get_FWHM(ham_window_128)
left256, half_maximum256, FWHM256 = get_FWHM(ham_window_256)

# print('FWHM of hamming window is respectively', FWHM64,FWHM128,FWHM256)
# x = np.linspace(0,511,512)
# plt.title('1D Point Spread Function of hamming window')
# plt.plot(x, np.abs(ham_window_64), color='green', label='ham_window_64: FWHM64=14')
# plt.plot(x[left64:left64+FWHM64], np.ones((FWHM64))*half_maximum64, color='green')
# plt.plot(x, np.abs(ham_window_128), color='red', label='ham_window_128: FWHM128=8')
# plt.plot(x[left128:left128+FWHM128], np.ones((FWHM128))*half_maximum128, color='red')
# plt.plot(x, np.abs(ham_window_256), color='yellow', label='ham_window_256: FWHM256=4')
# plt.plot(x[left256:left256+FWHM256], np.ones((FWHM256))*half_maximum256, color='yellow', ls = '--')
# plt.legend() # 显示图例
# plt.xlabel('frequency')
# plt.ylabel('amplitude')
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

