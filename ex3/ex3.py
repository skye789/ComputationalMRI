import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hamming
from numpy.ma import exp

# load matlab file
mat = scipy.io.loadmat('kdata_phase_error_severe.mat')
kspace_lack = mat['kdata']  # 512×288
mat0 = scipy.io.loadmat('kdata1.mat')
kspace_ori = mat0['kdata1']

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


rows, cols = np.shape(kspace_lack)  #512, 288
# real_cols = cols/9/16
zeros_pad = np.zeros((rows, rows - cols))
kspace_zeroPad = np.concatenate((kspace_lack, zeros_pad), axis=1)
image_zeroPad = ifft2c(kspace_zeroPad)


# kspace_halfF = np.fliplr(kspace_lack[:, :512 - cols].conjugate())  # flip left right
# kspace_halfF = np.concatenate((kspace_lack, kspace_halfF), axis=1)
# image_Hemitian = ifft2c(kspace_halfF)

kspace_halfF2 = np.flip(kspace_lack[:, :512 - cols].conjugate())  # flip left right and upside down
kspace_halfF2 = np.concatenate((kspace_lack, kspace_halfF2), axis=1)
image_Hemitian2 = ifft2c(kspace_halfF2)


# plt.figure()
# plt.subplot(121)
# plt.imshow(np.abs(np.real(image_zeroPad)), cmap='gray')
# plt.title('zero_padding image')
# plt.subplot(122)
# plt.imshow(np.abs(image_zeroPad), cmap='gray')
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

plt.figure()
plt.plot()
plt.imshow(phase, cmap='gray')
plt.title('Phase estimation from center kspace')
plt.show()


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
    filter = np.ones((512, 64))
    if ftype=="ramp":
        for i in range(64):
            filter[:, i] = -1 / 64 * i + 1
    elif ftype=="hamming":
        ham = hamming(128)
        filter = np.tile(ham[64:], (512, 1))

    kspace_ramp = np.copy(kspace)
    kspace_ramp[:, 224:288] = kspace_zeroPad[:, 224:288] * filter
    image_ramp = ifft2c(kspace_ramp)

    img_Margosian = np.real(exp((-1j) * phase) * image_ramp)

    return img_Margosian

img_Margosian_ramp = pf_margosian(kspace_zeroPad, N=9/16, ftype='ramp')
img_Margosian_ham = pf_margosian(kspace_zeroPad, N=9/16, ftype='hamming')

# plt.figure()
# plt.subplot(131)
# plt.imshow(np.abs(image_zeroPad), cmap='gray')
# plt.title('zero-padding image')
# plt.subplot(132)
# plt.imshow(np.abs(img_Margosian_ramp), cmap='gray')
# plt.title('image recon from Margosian(ramp)')
# plt.subplot(133)
# plt.imshow(np.abs(img_Margosian_ham), cmap='gray')
# plt.title('image recon from Margosian(hamming)')
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
    for _ in range(num_iter):
        image = ifft2c(kspace)
        image_POCS = np.abs(image) * exp(1j * phase)
        kspace_tmp = np.fft.fft2(image_POCS)
        kspace[:, 288:] = kspace_tmp[:, 288:]
    return image_POCS, kspace_tmp, kspace

image_POCS2, _, _ = pf_POCS(kspace_zeroPad, N=5, num_iter=2)
image_POCS4, _, _ = pf_POCS(kspace_zeroPad, N=5, num_iter=4)
image_POCS6, _, _ = pf_POCS(kspace_zeroPad, N=5, num_iter=6)
image_POCS8, kspace_tmp, kspacePOCS = pf_POCS(kspace_zeroPad, N=5, num_iter=8)
image_POCS10, _, _ = pf_POCS(kspace_zeroPad, N=5, num_iter=10)

# plt.figure()
# plt.plot()
# plt.subplot(121)
# plt.imshow(np.abs(image_POCS6), cmap='gray')
# plt.title('image reconstructed from POCS ')
# plt.subplot(122)
# plt.imshow(np.log((np.abs(kspacePOCS))), cmap='gray')
# plt.title('reconstructed kspace ')
# plt.show()

# plt.figure()
# plt.plot()
# plt.subplot(231)
# plt.imshow(np.abs(image_POCS2), cmap='gray')
# plt.title('image recon from POCS,iter=2 ')
# plt.subplot(232)
# plt.imshow((np.abs(image_POCS4)), cmap='gray')
# plt.title('image recon from POCS,iter=4 ')
# plt.subplot(233)
# plt.imshow((np.abs(image_POCS6)), cmap='gray')
# plt.title('image recon from POCS,iter=6 ')
# plt.subplot(234)
# plt.imshow((np.abs(image_POCS8)), cmap='gray')
# plt.title('image recon from POCS,iter=8 ')
# plt.subplot(235)
# plt.imshow((np.abs(image_POCS10)), cmap='gray')
# plt.title('image recon from POCS,iter=10 ')
# plt.show()



original_image = ifft2c(kspace_ori)

plt.figure()
plt.subplot(231)
plt.imshow(np.abs(np.abs(original_image)), cmap='gray')  #??用np.real不行， 用np.abs(np.real())出来的图像颜色变了
plt.title('original ')
plt.subplot(232)
plt.imshow(np.abs(image_zeroPad), cmap='gray')
plt.title('Zero padding')
plt.subplot(233)
plt.imshow(np.abs(image_Hemitian2), cmap='gray')
plt.title('Hermitian ')
plt.subplot(234)
plt.imshow(np.abs(img_Margosian_ramp), cmap='gray')
plt.title('Margosian(ramp)')
plt.subplot(235)
plt.imshow(np.abs(img_Margosian_ham), cmap='gray')
plt.title('Margosian(hamming)')
plt.subplot(236)
plt.imshow(np.abs(image_POCS10), cmap='gray')
plt.title('Pocs, iter=10')
plt.show()