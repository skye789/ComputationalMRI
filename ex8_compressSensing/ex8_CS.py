import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from lab8_basics import *

# load matlab file
mat = scipy.io.loadmat('data_lab8.mat')
kfull = mat['kfull']  #256×256
kacc = mat['kacc']  #256×256

def fft2c(img): #img原点在左上
    kspace = np.fft.fft2(img)
    kspace = np.fft.fftshift(kspace)
    return kspace

def ifft2c(kspace):
    image = np.fft.ifft2(kspace)
    img = np.fft.ifftshift(image)
    return img

#############################################################################
###### 1. Sparsity/compressibility of brain images using the wavelet transform:  ######
#############################################################################
def norm1_loss(img1, img2):
    norm1_loss = np.linalg.norm((img1 - img2), ord=1)
    return norm1_loss

origin = ifft2c(kfull)

def undersample_kspace(kfull, compress_factor):
    length = len(kfull)
    left = length//2-20
    right = length//2+20
    # ham_win = np.sqrt(np.outer(hamming(length), hamming(length * 2)))
    # weight = map(ham_win,0.5,1.5)
    # sample_mask = (np.random.rand(*kfull.shape)*weight) > (1 - 1/compress_factor)
    sample_mask = np.random.rand(*kfull.shape) > (1 - 1/compress_factor)
    sample_mask[left:right, left:right] = 1
    k_undersample = kfull*sample_mask
    return k_undersample,sample_mask

# k_undersample,sample_mask = undersample_kspace(kfull, compress_factor=2)
# img = ifft2c(k_undersample)
# img_wave, s = complex_wd2(img)
# recon_img = complex_wr2(img_wave, s)


# plt.imshow(sample_mask, cmap='gray')
# plt.title('sample_mask')

# plt.subplot(131)
# plt.imshow(np.abs(origin), cmap='gray')
# plt.title('Origin')
# plt.subplot(132)
# plt.imshow(np.abs(img_wave), cmap='gray')
# plt.title('DWT, kspce undersample sample 50%')
# plt.subplot(133)
# plt.imshow(np.abs(recon_img), cmap='gray')
# plt.title('Reconstruction')
# plt.show()

def denoise(img_wave,compress_factor,thres_type):
    m = np.sort(abs(img_wave.ravel()))[::-1]  #descending sort
    ndx = int(len(m) / compress_factor)
    # ndx = int(len(m) * 0.2)

    thr = m[ndx]
    if thres_type=='hard':
        img_wave_thr = HardT(img_wave,thr)   #img_wave * (abs(img_wave) > thr)
    elif thres_type=='soft':
        img_wave_thr = SoftT(img_wave, thr)
    else:
        raise ValueError("thres_type is wrong")
    return img_wave_thr


def map(data,MIN,MAX):
    """
    归一化映射到任意区间
    :param data: 数据
    :param MIN: 目标数据最小值
    :param MAX: 目标数据最小值
    :return:
    """
    d_min = np.max(data)    # 当前数据最大值
    d_max = np.min(data)    # 当前数据最小值
    return MIN + (MAX-MIN)/(d_max-d_min) * (data - d_min)


origin_map = map(np.abs(origin),0,1)
# plt.imshow(np.abs(origin_map), cmap='gray')
# plt.title('Original(after mapping)')
# plt.colorbar()
# plt.show()

def sparse(kfull, compress_factor):
    k_undersample, sample_mask = undersample_kspace(kfull, compress_factor=compress_factor)
    img_wave, s = complex_wd2(ifft2c(k_undersample))

    img_wave2 = denoise(img_wave, compress_factor=compress_factor, thres_type='soft')
    recon_img2 = complex_wr2(img_wave2, s)
    return img_wave2, recon_img2, sample_mask

# F = [2,3,5]
# F = [5, 10, 20]  # retain the largest 5%, 10%, 5% and 20% of the coefficients.
# img_index = np.arange(331,340)
# for i,f in enumerate(F):
#     img_wave2, recon_img2, _ = sparse(kfull, compress_factor=f)
#     recon_img2_map = map(np.abs(recon_img2),0,1)
#
#     plt.subplot(img_index[i])
#     plt.imshow(np.abs(img_wave2), cmap='gray')
#     plt.title('DWT,factor=' + str(f))
#     plt.axis('off')
#
#     plt.subplot(img_index[i + 3])
#     # plt.imshow(np.abs(recon_img2), cmap='gray')
#     plt.imshow(np.abs(recon_img2_map), cmap='gray')
#     plt.colorbar()
#     plt.title('CS(after mapping),factor=' + str(f))
#     plt.axis('off')
#
#     recon_error = np.abs(recon_img2_map) - np.abs(origin_map)
#     # recon_error = np.abs(recon_img2) - np.abs(origin)
#     plt.subplot(img_index[i + 6])
#     plt.imshow(map(np.abs(recon_error),0,0.1), cmap='gray')
#     # plt.imshow(np.abs(recon_error), cmap='gray')
#     plt.colorbar()
#     plt.title('Error, factor=' + str(f))
#     plt.axis('off')
#
#     print('Compress factor is %1.0f' % (f))
#     loss = norm1_loss(origin_map, recon_img2_map)
#     print('norm1_loss(afer mapping):', loss)
#
#     RMSE = np.sqrt(np.sum((np.abs(recon_img2_map) - np.abs(origin_map)) ** 2) / np.size(origin_map))
#     print("Root Mean Square Error(afer mapping) is %1.5f " %(RMSE))
#     print()
#
# plt.show()


#############################################################################
###### 2. Compressed sensing reconstruction using iterative soft thresholding:  ######
#############################################################################

def iterative_soft_thresholding(d,lam,compress_factor):
    '''''
    Input:
       d: adquired data, fully samply kspace
       m: original img 
    T and Ti are the forward and inverse sparsifying
    '''''
    m = ifft2c(d)  #initial solution
    iter_num = 50

    k_undersample, sample_mask = undersample_kspace(kfull, compress_factor=compress_factor)
    img_wave, s = complex_wd2(ifft2c(k_undersample))
    # img_wave2 = denoise(img_wave, compress_factor=compress_factor, thres_type='soft')
    # recon_img2 = complex_wr2(img_wave2, s)
    for i in range(iter_num):
        #enforce sparsity
        tmp, _ = complex_wd2(m)
        m = complex_wr2(SoftT(tmp, lam ), s)
        # enforce data consistency
        m = m - ifft2c((fft2c(m) * sample_mask - d) * sample_mask)
        # _, Tm, sample_mask = sparse(kfull, compress_factor)
        cost = np.linalg.norm(fft2c(m) * sample_mask - d, 2) + lam * np.linalg.norm(tmp, 1)
        print("cost in iteration  %1.0f is %1.5f" %(i, cost))

    return m


m = iterative_soft_thresholding(kfull,lam=0.2,compress_factor=5)
plt.imshow(np.abs(m),cmap='gray')
plt.show()


# num = np.count_nonzero(np.abs(kacc[:,1])>1e-8)
# print('sample line number ', num)
# acc_factor = len(kacc)/(len(kacc) - num)
# print('acc_factor ',acc_factor)


