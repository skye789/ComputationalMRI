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
sample_mask = np.random.rand(*kfull.shape) > (1 - 0.5)
k_undersample = kfull*sample_mask

img = ifft2c(kacc)
img_wave, s = complex_wd2(img)
recon_img = complex_wr2(img_wave, s)

# norm1_loss = norm1_loss(origin, recon_img)

# plt.subplot(131)
# plt.imshow(np.abs(origin), cmap='gray')
# plt.title('Origin')
# plt.subplot(132)
# plt.imshow(np.abs(img_wave), cmap='gray')
# plt.title('Daubechies wavelet transform')
# plt.subplot(133)
# plt.imshow(np.abs(recon_img), cmap='gray')
# plt.title('Reconstruction')
# plt.show()

def compress(img_wave,f,thres_type):
    '''''
    Input:
       f: compress factor 
    '''''
    m = np.sort(abs(img_wave.ravel()))[::-1]  #descending sort
    ndx = int(len(m) / f)
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


F = [5,10,20]  # retain the largest 5%, 10%, 5% and 20% of the coefficients.
img_index = np.arange(331,340)
for i,f in enumerate(F):
    img_wave2 = compress(img_wave, f=f, thres_type='soft')
    recon_img2 = complex_wr2(img_wave2, s)
    recon_img2_map = map(np.abs(recon_img2),0,1)

    plt.subplot(img_index[i])
    plt.imshow(np.abs(img_wave2), cmap='gray')
    plt.title('DWT,factor=' + str(f))
    plt.axis('off')

    plt.subplot(img_index[i + 3])
    # plt.imshow(map(np.abs(recon_img2),0,1), cmap='gray')
    plt.imshow(np.abs(recon_img2_map), cmap='gray')
    plt.colorbar()
    plt.title('CS(after mapping),factor=' + str(f))
    plt.axis('off')

    recon_error = np.abs(recon_img2_map) - np.abs(origin_map)
    # recon_error = np.abs(recon_img2) - np.abs(origin)
    plt.subplot(img_index[i + 6])
    plt.imshow(map(np.abs(recon_error),0,0.1), cmap='gray')
    # plt.imshow(np.abs(recon_error), cmap='gray')
    plt.colorbar()
    plt.title('Error, factor=' + str(f))
    plt.axis('off')

    print('Compress factor is %1.0f' % (f))
    loss = norm1_loss(origin_map, recon_img2_map)
    print('norm1_loss(afer mapping):', norm1_loss)

    RMSE = np.sqrt(np.sum((np.abs(recon_img2_map) - np.abs(origin_map)) ** 2) / np.size(origin_map))
    print("Root Mean Square Error(afer mapping) is %1.5f " %(RMSE))
    print()

# plt.show()


#############################################################################
###### 2. Compressed sensing reconstruction using iterative soft thresholding:  ######
#############################################################################

def iterative_soft_thresholding(d,m,lam,compression_ratio):
    '''''
    Input:
       d: fully samply kspace
       m: original img wave
       lam:lamda 
    T and Ti are the forward and inverse sparsifying
    '''''
    d = kfull
    m = img_wave
    sample_mask = np.random.rand(*m.shape) > (1 - compression_ratio)
    iter_num = 50

    # enforce sparsity:  m = Ti(SoftT(T(m), lam))
    m = compress(m, f=1 / lam, thres_type='soft')
    m = complex_wr2(m, s)

    for i in range(iter_num):
        # enforce data consistency
        m = m - ifft2c((fft2c(m) * sample_mask - d) * sample_mask)
        cost = np.linalg.norm(fft2c(m) * sample_mask - d, 2) + lam * np.linalg.norm(T(m), 1)
        print("cost in iteration  %1.0f is" %(cost))


