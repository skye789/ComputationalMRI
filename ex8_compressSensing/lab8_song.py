from warnings import filterwarnings
from scipy import io

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
# import torchkbnufft as tkbn
# import torch
from sklearn.metrics import mean_squared_error
from lab8_basics import complex_wd2, complex_wr2, SoftT, HardT, waveletShrinkage


# %%


def denoise(img_wave, compress_factor, thres_type):
    m = np.sort(abs(img_wave.ravel()))[::-1]  # descending sort
    ndx = int(len(m) / compress_factor)
    ndx = int(len(m) * 0.05)

    if thres_type == 'Hard':
        img_wave_thr = HardT(img_wave, thr)  # img_wave * (abs(img_wave) > thr)
    elif thres_type == 'soft':
        img_wave_thr = SoftT(img_wave, thr)
    else:
        raise ValueError("thres_type is wrong")
    return img_wave_thr


### load data
kdata1 = scipy.io.loadmat('data_lab8.mat')
# print(kdata1)
kfull = kdata1['kfull']
kacc = kdata1['kacc']
print(kacc.shape)


# %%

def undersample_kspace(kfull, compress_factor):
    length = len(kfull)
    left = length // 2 - 20
    right = length // 2 + 20
    # ham_win = np.sqrt(np.outer(hamming(length), hamming(length * 2)))
    # weight = map(ham_win,0.5,1.5)
    # sample_mask = (np.random.rand(*kfull.shape)*weight) > (1 - 1/compress_factor)
    sample_mask = np.random.rand(*kfull.shape) > (1 - 1 / compress_factor)
    sample_mask[left:right, left:right] = 1
    k_undersample = kfull * sample_mask
    return k_undersample, sample_mask


### e1
# org_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kfull)))
# k_undersample, sample_mask = undersample_kspace(kfull, compress_factor=3)
# img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k_undersample)))
#
# wave_rep, s = complex_wd2(img)
# # img1 = HardT(img1,0.01)
# wave_rep = denoise(wave_rep, compress_factor=3, thres_type='Hard')
#
# rec_img = complex_wr2(wave_rep, s)
#
# diff = np.sum(np.abs(org_img - rec_img))


# # %%
#
# plt.figure()
# plt.imshow(np.abs(org_img), cmap='gray')
# plt.title('org_img')
# plt.axis('off')
# # plt.savefig('./figs/NUFFT_ls_combined_im.png')
# plt.show()
#
# # %%
#
# plt.figure()
# plt.imshow(np.abs(wave_rep), cmap='gray')
# plt.title('wavelet representation')
# plt.axis('off')
# # plt.savefig('./figs/NUFFT_ls_combined_im.png')
# plt.show()

# %%

# plt.figure()
# plt.imshow(np.abs(rec_img), cmap='gray')
# plt.title('rec_img_factor=3')
# plt.axis('off')
# plt.show()
#
# plt.figure()
# plt.imshow(np.abs(org_img) - np.abs(rec_img), cmap='gray')
# plt.title('err_img_factor=10')
# plt.axis('off')
# plt.show()
#
# RMSE = np.sqrt(np.sum((np.abs(rec_img) - np.abs(org_img)) ** 2) / np.size(org_img))
# print("Root Mean Square Error of factor 10 is %1.5f " % (RMSE))


# tv = np.sum(np.abs(np.diff(org_img)))
# tv = tv + np.sum(np.abs(np.diff(org_img, axis=0)))
# print('TV for org_img : ' + str(tv))
#
# tv = np.sum(np.abs(np.diff(rec_img)))
# tv = tv + np.sum(np.abs(np.diff(rec_img, axis=0)))
# print('TV for rec_img : ' + str(tv))
#
# diff_rec = np.diff(rec_img) + np.diff(rec_img, 1)
# diff_org = np.diff(org_img) + np.diff(org_img, 1)
# l1 = np.linalg.norm(diff_org, 1)
# l11 = np.linalg.norm(diff_rec, 1)
#
# print('l1norm for org_img : ' + str(l1))
# print('l1norm for rec_img : ' + str(l11))
#
# ### ex1.2
def fft2c(img):  # img原点在左上
    kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
    return kspace


def ifft2c(kspace):
    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))
    return img


def sparse(k_undersample, compress_factor):
    # k_undersample, sample_mask = undersample_kspace(kfull, compress_factor=compress_factor)
    img_wave, s = complex_wd2(k_undersample)

    img_wave2 = denoise(img_wave, compress_factor=compress_factor, thres_type='soft')
    recon_img2 = complex_wr2(img_wave2, s)
    return img_wave2, recon_img2


def iterative_soft_thresholding(d, lam, compress_factor):
    '''''
    Input:
       d: adquired data, fully samply kspace
       m: original img 
    T and Ti are the forward and inverse sparsifying
    '''''
    iter_num = 300
    # kundersample, sample_mask = undersample_kspace(kfull, compress_factor)
    kundersample = kacc
    sample_mask = mask
    m = ifft2c(kundersample)
    d = kacc
    Tm = np.zeros_like(m)

    for i in range(iter_num):
        # enforce sparsity

        cost = np.linalg.norm(fft2c(m) * sample_mask - d, 2) + lam * np.linalg.norm(Tm, 1)
        print("cost in iteration  %1.0f is %1.5f" % (i, cost))
        m = m - ifft2c((fft2c(m) * sample_mask - d) * sample_mask)
        Tm, m = sparse(m, compress_factor)

    return m


mask = np.zeros_like(kfull)
print(mask.shape)
idx = np.where(abs(kacc) > 0)
mask[idx] = 1
# print(mask)
m = iterative_soft_thresholding(kfull, lam=0.02, compress_factor=1.8)
plt.imshow(np.abs(m), cmap='gray')
plt.title('l=0.5%')
plt.show()
