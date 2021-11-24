import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hamming
from numpy.ma import exp

# load matlab file
mat = scipy.io.loadmat('kdata_phase_error_severe.mat')
kspace_lack = mat['kdata']  # 512×288
mat0 = scipy.io.loadmat('kdata1.mat')
kspace = mat0['kdata1']

#######################################################
###### 1. Hermitian symmetry reconstructed image ######
#######################################################
def fft2c(img):
    kspace = np.fft.fft2(img)
    kspace = np.fft.fftshift(kspace)  #????????????
    return kspace


def ifft2c(kspace):
    image = np.fft.ifft2(kspace)
    img = np.fft.ifftshift(image)
    return img


rows, cols = np.shape(kspace_lack)
zeros_pad = np.zeros((rows, 512 - cols))
kspace_zeroPad = np.concatenate((kspace_lack, zeros_pad), axis=1)
image_zeroPad = ifft2c(kspace_zeroPad)
# a = kspace_lack[0,255]
# b = kspace_lack[-1,256]
# c = kspace_lack[0,256]
# print(a,b,c) #b or c is not conjungate complext of a?????

kspace_halfF = np.fliplr(kspace_lack[:, :512 - cols].conjugate())  # flip left right
kspace_halfF = np.concatenate((kspace_lack, kspace_halfF), axis=1)
image_Hemitian = ifft2c(kspace_halfF)

# kspace_halfF2 = np.flip(kspace_lack[:, :512 - cols].conjugate())  # flip left right and upside down
# kspace_halfF2 = np.concatenate((kspace_lack, kspace_halfF2), axis=1)
# image_Hemitian2 = ifft2c(kspace_halfF2)


# plt.figure()
# plt.subplot(121)
# plt.imshow(np.abs(image_zeroPad), cmap='gray')
# plt.title('zero_padding image')
# plt.subplot(122)
# plt.imshow(np.abs(image_Hemitian), cmap='gray')
# plt.title('Hermitian symmetry reconstructed image')
# plt.show()


###############################
######2.Phase estimation ######
###############################
def phase_sym_ham(kspace_asym):
    '''''
    purpose:  to estimate the phase of an image from a symmetric region at the 
              center of k-space
    kspace: asymmetric k-space data
    '''''
    kspace_cen = kspace_asym[:, 256 - 32:256 + 32]  # 512×64
    kspace_cen_pad = np.pad(kspace_cen, ((0, 0), (224, 224)), 'constant', constant_values=(0, 0))
    ham_win = np.sqrt(np.outer(hamming(512), hamming(64)))
    ham_win_pad = np.pad(ham_win, ((0, 0), (224, 224)), 'constant', constant_values=(0, 0))
    img_k_cen = ifft2c(kspace_cen_pad * ham_win_pad)
    phase = np.angle(img_k_cen)

    return img_k_cen, phase

img_k_cen, phase= phase_sym_ham(kspace_lack)

# plt.figure()
# plt.plot()
# plt.imshow(phase, cmap='gray')
# plt.title('Phase estimation from center kspace')
# plt.show()


#################################
###### 3. Margosian method ######
#################################
def pf_margosian(kspace, N, ftype):
    '''''
    purpose: reconstruct partial Fourier MRI data using the Margosian method
    kspace: asymmetric k-space data
    N: target size of the reconstructed PF dimension
    ftype: k-space filter ('ramp' or 'hamming')
    '''''
    filter = np.ones((512, 32))
    if ftype=="ramp":
        for i in range(32):
            filter[:, i] = -1 / 32 * i + 1
    elif ftype=="hamming":
        ham = hamming(64)
        filter = np.tile(ham[32:], (512, 1))

    kspace_ramp = np.copy(kspace)
    kspace_ramp[:, 256:288] = kspace_zeroPad[:, 256:288] * filter
    image_ramp = ifft2c(kspace_ramp)

    img_Margosian = exp((-1j) * phase) * image_ramp
    return img_Margosian

img_Margosian_ramp = pf_margosian(kspace_zeroPad, N=1, ftype='ramp')
img_Margosian_ham = pf_margosian(kspace_zeroPad, N=1, ftype='hamming')

# plt.figure()
# plt.plot()
# plt.imshow(img_Margosian_ham, cmap='gray')
# plt.title('Phase estimation from center kspace')
# plt.show()


#################################
###### 4. POCS method ###########
#################################
def pf_POCS(kspace, N, num_iter):
    '''''
    purpose: reconstruct partial Fourier MRI data using the POCS method
    kspace: asymmetric k-space data
    N: target size of the reconstructed PF dimension
    num_iter: number of iteration
    '''''
    phase_POSC = exp(1j * phase)
    for _ in range(num_iter):
        image = ifft2c(kspace)
        image_POCS = np.abs(image) * phase_POSC
        # kspace_tmp = fft2c(image_POCS)
        kspace_tmp = np.fft.fft2(image_POCS)
        kspace[:, 288:] = kspace_tmp[:, 288:]
    return kspace,image_POCS, kspace_tmp

kspaceP,image_POCS, kspace_tmp = pf_POCS(kspace_zeroPad, N=5, num_iter=5)

plt.figure()
plt.plot()
plt.subplot(121)
plt.imshow(np.abs(image_POCS), cmap='gray')
plt.subplot(122)
plt.imshow(np.log((np.abs(kspace_tmp))), cmap='gray')
plt.show()





original_image = ifft2c(kspace)

# plt.figure()
# plt.subplot(231)
# plt.imshow(np.abs(np.abs(original_image)), cmap='gray')  #??用np.real不行， 用np.abs(np.real())出来的图像颜色变了
# plt.title('original ')
# plt.subplot(232)
# plt.imshow(np.abs(image_Hemitian), cmap='gray')
# plt.title('Hermitian ')
# plt.subplot(234)
# plt.imshow(np.abs(img_Margosian_ramp), cmap='gray')
# plt.title('Margosian(ramp)')
# plt.subplot(235)
# plt.imshow(np.abs(img_Margosian_ham), cmap='gray')
# plt.title('Margosian(hamming)')
# plt.subplot(236)
# plt.imshow(np.abs(image_POCS), cmap='gray')
# plt.title('Pocs ')
# plt.show()