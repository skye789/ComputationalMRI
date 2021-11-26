import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hamming
from numpy.ma import exp

# load matlab file
mat = scipy.io.loadmat('kdata_phase_error_severe.mat')
kspace = mat['kdata']  # 512×288
mat0 = scipy.io.loadmat('kdata1.mat')
kspace_ori = mat0['kdata1']

#######################################################
###### 1. Hermitian symmetry reconstructed image ######
#######################################################
def fft2c(image):
    image = np.fft.fftshift(image)
    kspace = np.fft.fft2(image)
    kspace = np.fft.ifftshift(kspace)
    return kspace

def ifft2c(kspace):
    kspace = np.fft.ifftshift(kspace)
    image = np.fft.ifft2(kspace)
    image = np.fft.fftshift(image)
    return image

rows, cols = np.shape(kspace)  #512, 288
recon_cols = int(cols/9*16)  # 512
zeros_pad = np.zeros((rows, recon_cols - cols))
kspace_zeroPad = np.concatenate((kspace, zeros_pad), axis=1)
image_zeroPad = ifft2c(kspace_zeroPad)


# kspace_halfF = np.fliplr(kspace[:, :512 - cols].conjugate())  # flip left right
# kspace_halfF = np.concatenate((kspace, kspace_halfF), axis=1)
# image_Hemitian = ifft2c(kspace_halfF)

kspace_halfF2 = np.flip(kspace[:, :recon_cols - cols].conjugate())  # flip left right and upside down
kspace_halfF2 = np.concatenate((kspace, kspace_halfF2), axis=1)
image_Hemitian2 = ifft2c(kspace_halfF2)


# plt.figure()
# plt.subplot(121)
# plt.imshow(np.abs(np.real(image_zeroPad)), cmap='gray')
# plt.title('zero_padding image')
# plt.subplot(122)
# plt.imshow(np.abs(image_Hemitian2), cmap='gray')
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
    num_row, num_col = kspace_asym.shape #512, 288
    num_realCol = int(num_col/9*16) #512
    asy_cols = num_col - num_realCol//2  #32
    kspace_cen = kspace_asym[:, num_realCol//2 - asy_cols:num_realCol//2 + asy_cols]  # 512×64
    kspace_cen_pad = np.pad(kspace_cen, ((0, 0), (224, 224)), 'constant', constant_values=(0, 0))
    ham_win = np.sqrt(np.outer(hamming(num_realCol), hamming(asy_cols*2)))
    ham_win_pad = np.pad(ham_win, ((0, 0), (224, 224)), 'constant', constant_values=(0, 0))
    img_k_cen = ifft2c(kspace_cen_pad * ham_win_pad)
    phase = np.angle(img_k_cen)

    return img_k_cen, phase

img_k_cen, phase= phase_sym_ham(kspace)

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
    num_row, num_col = kspace.shape  #512, 288
    num_realCol = int(num_col/N) #512
    asy_cols = num_col - num_realCol//2  #32

    zeros_pad = np.zeros((rows, num_realCol - num_col))
    kspace_zeroPad = np.concatenate((kspace, zeros_pad), axis=1)

    filter = np.ones((num_row, asy_cols*2))
    if ftype=="ramp":
        for i in range( asy_cols*2):
            filter[:, i] = -1 / (asy_cols*2) * i + 1
    elif ftype=="hamming":
        ham = hamming(asy_cols*4)
        filter = np.tile(ham[asy_cols*2:], (num_row, 1))

    kspace_fil = np.copy(kspace_zeroPad)
    kspace_fil[:, num_realCol//2-asy_cols:num_realCol//2+asy_cols] = \
        kspace_zeroPad[:, num_realCol//2-asy_cols:num_realCol//2+asy_cols] * filter
    image_ramp = ifft2c(kspace_fil)

    img_Margosian = np.real(exp((-1j) * phase) * image_ramp)

    return img_Margosian

img_Margosian_ramp = pf_margosian(kspace, N=9/16, ftype='ramp')
img_Margosian_ham = pf_margosian(kspace, N=9/16, ftype='hamming')

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
    num_row, num_col = kspace.shape  #512, 288
    num_realCol = int(num_col/N) #512
    asy_cols = num_col - num_realCol//2  #32

    zeros_pad = np.zeros((rows, num_realCol - num_col))
    kspace_zeroPad = np.concatenate((kspace, zeros_pad), axis=1)

    for _ in range(num_iter):
        image = ifft2c(kspace_zeroPad)
        image_POCS = np.abs(image) * exp(1j * phase)
        kspace_tmp = fft2c(image_POCS)
        kspace_zeroPad[:, num_realCol//2+asy_cols:] = kspace_tmp[:, num_realCol//2+asy_cols:]
    return image_POCS, kspace_tmp, kspace_zeroPad

image_POCS2, _, _ = pf_POCS(kspace, N=9/16, num_iter=2)
image_POCS4, _, _ = pf_POCS(kspace, N=9/16, num_iter=4)
image_POCS6, _, _ = pf_POCS(kspace, N=9/16, num_iter=6)
image_POCS8, kspace_tmp, kspacePOCS = pf_POCS(kspace, N=9/16, num_iter=8)
image_POCS10, _, _ = pf_POCS(kspace, N=9/16, num_iter=10)

# plt.figure()
# plt.plot()
# plt.subplot(121)
# plt.imshow(np.abs(image_POCS6), cmap='gray')
# plt.title('image reconstructed from POCS ')
# plt.subplot(122)
# plt.imshow(np.log((np.abs(kspacePOCS))), cmap='gray')
# plt.title('reconstructed kspace ')
# plt.show()

plt.figure()
plt.plot()
plt.subplot(231)
plt.imshow(np.abs(image_POCS2), cmap='gray')
plt.title('image recon from POCS,iter=2 ')
plt.subplot(232)
plt.imshow((np.abs(image_POCS4)), cmap='gray')
plt.title('image recon from POCS,iter=4 ')
plt.subplot(233)
plt.imshow((np.abs(image_POCS6)), cmap='gray')
plt.title('image recon from POCS,iter=6 ')
plt.subplot(234)
plt.imshow((np.abs(image_POCS8)), cmap='gray')
plt.title('image recon from POCS,iter=8 ')
plt.subplot(235)
plt.imshow((np.abs(image_POCS10)), cmap='gray')
plt.title('image recon from POCS,iter=10 ')
plt.show()



original_image = ifft2c(kspace_ori)

# plt.figure()
# plt.subplot(231)
# plt.imshow(np.abs(original_image), cmap='gray')
# plt.title('original ')
# plt.subplot(232)
# plt.imshow(np.abs(image_zeroPad), cmap='gray')
# plt.title('Zero padding')
# plt.subplot(233)
# plt.imshow(np.abs(image_Hemitian2), cmap='gray')
# plt.title('Hermitian ')
# plt.subplot(234)
# plt.imshow(np.abs(img_Margosian_ramp), cmap='gray')
# plt.title('Margosian(ramp)')
# plt.subplot(235)
# plt.imshow(np.abs(img_Margosian_ham), cmap='gray')
# plt.title('Margosian(hamming)')
# plt.subplot(236)
# plt.imshow(np.abs(image_POCS10), cmap='gray')
# plt.title('Pocs, iter=10')
# plt.show()
a = np.abs(img_Margosian_ham)-np.real(original_image)
b = np.abs(image_POCS10)-np.abs(original_image)
##show the residual artifact
plt.figure()
plt.subplot(231)
plt.imshow(np.abs(original_image), cmap='gray')
plt.title('original ')
plt.subplot(232)
plt.imshow(np.abs(image_zeroPad)-np.abs(original_image), cmap='gray')
plt.title('Residual artifact from Zero padding')
plt.subplot(233)
plt.imshow(np.abs(image_Hemitian2)-np.abs(original_image), cmap='gray')
plt.title('Residual artifact from Hermitian ')
plt.subplot(234)
plt.imshow(np.abs(img_Margosian_ramp)-np.abs(original_image), cmap='gray')
plt.title('Residual artifact from Margosian(ramp)')
plt.subplot(235)
plt.imshow(np.abs(img_Margosian_ham)-np.abs(original_image), cmap='gray')
plt.title('Residual artifact from Margosian(hamming)')
plt.subplot(236)
plt.imshow(np.abs(image_POCS10)-np.abs(original_image), cmap='gray')
plt.title('Residual artifact from Pocs, iter=10')
plt.show()