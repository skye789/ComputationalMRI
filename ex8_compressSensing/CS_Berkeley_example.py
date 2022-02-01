
#refer to:   https://inst.eecs.berkeley.edu/~ee123/sp16/hw/hw9_Compressed_Sensing.html
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

import pywt
plt.rcParams['figure.figsize'] = (16, 16)

# load matlab file
mat = scipy.io.loadmat('data_lab8.mat')
kfull = mat['kfull']  #256×256
kacc = mat['kacc']  #256×256



def ifft2c(kspace):
    image = np.fft.ifft2(kspace)
    img = np.fft.ifftshift(image)
    return img

im = ifft2c(kfull)

def imshowgray(im, vmin=None, vmax=None):
    plt.imshow(im, cmap=plt.get_cmap('gray'), vmin=vmin, vmax=vmax)


def wavMask(dims, scale):
    sx, sy = dims
    res = np.ones(dims)
    NM = np.round(np.log2(dims))
    for n in range(int(np.min(NM) - scale + 2) // 2):
        res[:int(np.round(2 ** (NM[0] - n))), :int(np.round(2 ** (NM[1] - n)))] = \
            res[:int(np.round(2 ** (NM[0] - n))), :int(np.round(2 ** (NM[1] - n)))] / 2
    return res


def imshowWAV(Wim, scale=1):
    plt.imshow(np.abs(Wim) * wavMask(Wim.shape, scale), cmap=plt.get_cmap('gray'))


def coeffs2img(LL, coeffs):
    LH, HL, HH = coeffs
    return np.vstack((np.hstack((LL, LH)), np.hstack((HL, HH))))


def unstack_coeffs(Wim):
    L1, L2 = np.hsplit(Wim, 2)
    LL, HL = np.vsplit(L1, 2)
    LH, HH = np.vsplit(L2, 2)
    return LL, [LH, HL, HH]


def img2coeffs(Wim, levels=4):
    LL, c = unstack_coeffs(Wim)
    coeffs = [c]
    for i in range(levels - 1):
        LL, c = unstack_coeffs(LL)
        coeffs.insert(0, c)
    coeffs.insert(0, LL)
    return coeffs


def dwt2(im):
    coeffs = pywt.wavedec2(im, wavelet='db4', mode='per', level=4)
    Wim, rest = coeffs[0], coeffs[1:]
    for levels in rest:
        Wim = coeffs2img(Wim, levels)
    return Wim


def idwt2(Wim):
    coeffs = img2coeffs(Wim, levels=4)
    return pywt.waverec2(coeffs, wavelet='db4', mode='per')


# img_wave = dwt2(im)
# recon_img = idwt2(img_wave)

# plt.subplot(1,3,1)
# imshowgray(np.abs(im))
# plt.title('Original')
#
# plt.subplot(1,3,2)
# imshowWAV(img_wave)
# plt.title('DWT')
#
# plt.subplot(1,3,3)
# imshowgray(np.abs(recon_img))
# plt.title('Reconstruction')
# plt.show()
#
# print 'Reconstruction error:', np.linalg.norm(im - im2)


def compress(img_wave,f):
    '''''
    Input:
       f: compress factor 
    '''''
    m = np.sort(abs(img_wave.ravel()))[::-1]  #descending sort
    ndx = int(len(m) / f)
    thr = m[ndx]
    img_wave_thr = img_wave * (abs(img_wave) > thr)
    return img_wave_thr

def norm1_loss(img1, img2):
    norm1_loss = np.linalg.norm((img1 - img2), ord=1)
    print('Reconstruction error:', norm1_loss)
    return norm1_loss

img_wave = dwt2(im)
recon_img = idwt2(img_wave)

img_wave2 = compress(img_wave,f=10)
recon_img2 = idwt2(img_wave2)
norm1_loss = norm1_loss(im, recon_img2)

plt.imshow(np.abs(recon_img2), cmap='gray')
plt.show()