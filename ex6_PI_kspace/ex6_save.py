import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import fractional_matrix_power

# from grappaR2K2x3 import grappaR2K2x3

# load matlab file
mat = scipy.io.loadmat('data_brain_8coils.mat')
c_coil_sen = mat['c']  # coil sensitivity maps (256×256×8)
fs_kdata = mat['d']  # fully-sampled k-space 256×256×8  [PE,FE,channels],
noise = mat['n']  # noise-only scan (256×8)

nx, ny, nc = np.shape(fs_kdata)


def ifft2c(kspace):
    image = np.fft.ifft2(kspace)
    img = np.fft.ifftshift(image)
    return img


# least square with_noiseCorrelation (match filter)
def least_square_with_noiseCorrelation(m_coil_img, c_coil_sen, noise_cov):
    '''''
    Input:    
        m_coil_img: [Nx,Ny,Nc]
        c_coil_sen: [Nx,Ny,Nc]
        noise_cov: [Nc,Nc]
    Return:
        img_least_square: least square combination image through coils[Nx,Ny]
    '''''
    # scipy.linalg support fractional matrix power
    n_cov_inv_half = fractional_matrix_power(noise_cov, -1 / 2)

    # correct coil img and coil sensitivity with half inverse of noise covariance matrix
    m_w_coil_img = np.zeros_like(m_coil_img)
    c_w_coil_sen = np.zeros_like(c_coil_sen)
    for i in range(nc):  # number of chanel
        for j in range(nc):
            m_w_coil_img[:, :, i] += n_cov_inv_half[i, j] * m_coil_img[:, :, j]
            c_w_coil_sen[:, :, i] += n_cov_inv_half[i, j] * c_coil_sen[:, :, j]

    m_w_star_coil_img = m_w_coil_img.conjugate()
    f_img = np.zeros_like(m_w_star_coil_img)
    for i in range(nc):
        f_img[..., i] = m_w_star_coil_img[..., i] * c_w_coil_sen[..., i]

    img_least_square = np.sum(f_img, axis=2)  # 256×256
    coil_sen2D = np.linalg.norm(c_w_coil_sen, axis=2) + 10 ** -5  # 256×256
    img_least_square = img_least_square / coil_sen2D

    return img_least_square, coil_sen2D


m_coil_img = np.zeros_like(fs_kdata)
for i in range(nc):
    m_coil_img[..., i] = ifft2c(fs_kdata[..., i])
noise_cov = np.cov(np.transpose(noise.conjugate()))  # 8×8
img_least_square_noiseCorrelation, coil_sen2D_noiseCorrelation = least_square_with_noiseCorrelation(m_coil_img,
                                                                                                    c_coil_sen,
                                                                                                    noise_cov)


# least square(match filter)
def least_square(m_coil_img, c_coil_sen):
    m_star_coil_img = m_coil_img.conjugate()
    f_img = np.zeros_like(c_coil_sen)
    for i in range(nc):  # number of chanel
        f_img[..., i] = m_star_coil_img[..., i] * c_coil_sen[..., i]
    img_sum = np.sum(f_img, axis=2)  # 256×256

    coil_sen2D = np.linalg.norm(c_coil_sen, axis=2) + 10 ** -5  # 256×256
    img_least_square = img_sum / coil_sen2D
    return img_least_square, coil_sen2D


def get_acs(num_cenLine, fs_kdata):
    acs = fs_kdata[(nx - num_cenLine) // 2: (nx + num_cenLine) // 2]
    return acs


def get_zf_kspace(R, fs_kdata):
    """""
    Input:
        R: acceleration factor
        fs_kdata: fully sampled kdata. (nx, ny, nc)
    Output:
        zf_kspace:fill every other line to zero, if R=2 (nx, ny, nc)
    """""
    zf_kspace = np.zeros_like(fs_kdata)
    for i in range(nx):
        if i % R == 0:  # even
            zf_kspace[i] = fs_kdata[i]
    return zf_kspace


def get_Img(inpo_ksp):
    """""
    combine kspace in diff coil and reconstruct image
    """""
    m_coil_img = np.zeros_like(inpo_ksp)
    for i in range(nc):
        m_coil_img[..., i] = ifft2c(inpo_ksp[..., i])
    img_least_square, _ = least_square(m_coil_img, c_coil_sen)
    return img_least_square


#############################################################
###### 1. Simple GRAPPA reconstruction  ######
#############################################################
# zf_kspace = get_zf_kspace(R=2, fs_kdata=fs_kdata)
# acs = get_acs(num_cenLine=24, fs_kdata=fs_kdata)
# inpo_ksp, _ = grappaR2K2x3(zf_kspace, acs, flag_acs = False)
# img_R2K2x3 = get_Img(inpo_ksp)

# plt.imshow(np.log(np.abs(zf_kspace[...,0])))
# plt.imshow(np.abs(img_R2K2x3), cmap='gray')
# plt.title('GRAPPA, Ry=2, kernel2×3')
# plt.show()

#############################################################
###### 2. Modify GRAPPA reconstruction  ######
#############################################################
"""""
def extract(acs, k_shp, omit=None, ):
    nxacs, nyacs, nc = acs.shape
    nxk, nyk = k_shp
    nb = int((nxacs-(nxk-1)/2) * (nyacs - 2))
    nk = nxk * nyk  # kernel size, 12

    # initial source matrix
    src = np.zeros((nb, nc * nk), dtype=np.complex64)

    # initial target matrix
    targ = np.zeros((nb, nc), dtype=np.complex64)

    src_idx = 0
    # dx_list = np.linspace(-1-(nxk/2-1)*2, 1+(nxk/2-1)*2, nxk, endpoint=True) # [-3,-1,1,3], if nxk=4
    dx_list = [-3,-1,1,3] #if nxk=4
    dy_list = np.arange(nyk) - 1
    for yi in range(1, nyacs - 1):
        for xi in range((nxk-1), nxacs - (nxk-1)):  # for循环所有target点(nb)

            block = np.array([[acs[xi + int(dx), yi + dy] for dx in dx_list] for dy in dy_list]).flatten()

            src[src_idx] = block
            targ[src_idx] = acs[xi, yi]

            src_idx += 1

    return src, targ


def interp(zp_kspace, ws, k_shp, omit):
    #zp_kspace, (nx + 5, ny + 2, nc)
    nxzpk, nyzpk, nc = zp_kspace.shape
    nxk, nyk = k_shp
    nb = int((nxzpk - (nxk - 1) / 2) * (nyzpk - 2))

    # initial target matrix
    targ = np.zeros((nb, nc), dtype=np.complex64)

    # for循环所有target点(nb)
    interpolated = np.array(zp_kspace)
    src_idx = 0
    dx_list = np.linspace(-1-(nxk/2-1)*2, 1+(nxk/2-1)*2, nxk, endpoint=True)
    dx_list = [-3,-1,1,3]
    dy_list = np.arange(nyk) - 1
    for yi in range(1, nyzpk - 1):
        for xi in range((nxk-1), nxzpk - (nxk-1)):  # for循环所有target点(nb)
            if omit is not None:
                if xi % omit == 0:
                    continue

            block = np.array([[zp_kspace[xi + int(dx), yi + dy] for dx in dx_list] for dy in dy_list]).flatten() # 2×3×8
            interpolated[xi , yi] = np.dot(block, ws)  # ws:
            targ[src_idx] = zp_kspace[xi, yi]

            src_idx += 1

    return interpolated, zp_kspace, targ


## R=2
def grappa4x3(kdata, acs, flag_acs, k_shp, R=2):
    nx, ny, nc = kdata.shape
    nxacs, nyacs, _ = acs.shape

    # get source and target in acs region, to calculate weight
    src, targ = extract(acs, k_shp=k_shp)
    ws = np.dot(np.linalg.pinv(src), targ)  # different from original implementation

    zp_up = nxacs-2
    zp_dn = nxacs-1
    zp_kdata = np.zeros((nx + zp_up+zp_dn, ny + 2, nc), np.complex64)
    zp_kdata[zp_up:nx+zp_up, 1:ny + 1, :] = kdata

    # interpolation
    interpolated, _, _ = interp(zp_kdata, ws, k_shp=k_shp, omit=2)
    interpolated = interpolated[zp_up:nx+zp_up, 1:ny + 1, :]

    if flag_acs:
        interpolated[int(nx / 2 - nxacs / 2):int(nx / 2 + nxacs / 2)] = acs

    return interpolated, zp_kdata
"""


def extract(acs, k_shp, R, omit=None):
    global dx_list
    nxacs, nyacs, nc = acs.shape
    nxk, nyk = k_shp  # 4×3
    start = (R + R // 2) * 2
    nb = int((nxacs - start * 2) * (nyacs - 2))  # 10=(R+1)*2
    nk = nxk * nyk  # kernel size, 12

    # initial source matrix
    src = np.zeros((nb, nc * nk), dtype=np.complex64)

    # initial target matrix
    targ = np.zeros((nb, nc), dtype=np.complex64)

    src_idx = 0
    # dx_list = np.linspace(-1-(nxk/2-1)*2, 1+(nxk/2-1)*2, nxk, endpoint=True) # [-3,-1,1,3], if nxk=4
    if R == 2:
        dx_list = np.array([-3, -1, 1, 3])
    elif R == 4:
        dx_list = np.array([-6, -2, 2, 6])
    dy_list = np.arange(nyk) - 1
    for yi in range(1, nyacs - 1):
        for xi in range(start, nxacs - start):  # for循环所有target点(nb)

            block = np.array([[acs[xi + int(dx), yi + dy] for dx in dx_list] for dy in dy_list]).flatten()

            src[src_idx] = block
            targ[src_idx] = acs[xi, yi]

            src_idx += 1

    return src, targ


def interp(zp_kspace, ws, k_shp, omit):
    R = omit
    nxzpk, nyzpk, nc = zp_kspace.shape
    nxk, nyk = k_shp
    start = R + R // 2
    nb = int((nxzpk - start * 2) * (nyzpk - 2))  # 10=(R+1)*2
    # initial target matrix
    targ = np.zeros((nb, nc), dtype=np.complex64)

    # for循环所有target点(nb)
    interpolated = np.array(zp_kspace)
    src_idx = 0
    dx_list = np.array([-6, -2, 2, 6])
    dy_list = np.arange(nyk) - 1
    for yi in range(1, nyzpk - 1):
        for xi in range(start, nxzpk - start):  # for循环所有target点(nb),
            if (xi - start - 1) % omit == 0:
                print(xi)
                continue
            elif (xi - start - 1) % omit == 1:
                dx_list = np.array([-6, -2, 2, 6]) + 1
            elif (xi - start - 1) % omit == 2:
                dx_list = np.array([-6, -2, 2, 6])
            elif (xi - start - 1) % omit == 3:
                dx_list = np.array([-6, -2, 2, 6]) - 1

            block = np.array(
                [[zp_kspace[xi + int(dx), yi + dy] for dx in dx_list] for dy in dy_list]).flatten()  # 2×3×8
            interpolated[xi, yi] = np.dot(block, ws)  # ws:
            targ[src_idx] = zp_kspace[xi, yi]

            src_idx += 1

    return interpolated, zp_kspace, targ


def grappa4x3(kdata, acs, flag_acs, k_shp, R):
    nx, ny, nc = kdata.shape
    nxacs, nyacs, _ = acs.shape

    # get source and target in acs region, to calculate weight
    src, targ = extract(acs, k_shp=k_shp, R=R)
    ws = np.dot(np.linalg.pinv(src), targ)

    zp_up = R + R // 2 - 1
    zp_dn = R + R // 2
    zp_kdata = np.zeros((nx + zp_up + zp_dn, ny + 2, nc), np.complex64)
    zp_kdata[zp_up:nx + zp_up, 1:ny + 1, :] = kdata

    # interpolation
    interpolated, _, _ = interp(zp_kdata, ws, k_shp=k_shp, omit=R)
    interpolated = interpolated[zp_up:nx + zp_up, 1:ny + 1, :]

    if flag_acs:
        interpolated[int(nx / 2 - nxacs / 2):int(nx / 2 + nxacs / 2)] = acs

    return interpolated, zp_kdata


zf_kspace = get_zf_kspace(R=4, fs_kdata=fs_kdata)
acs = get_acs(48, fs_kdata=fs_kdata)
inpo_ksp, _ = grappa4x3(zf_kspace, acs, flag_acs=False, k_shp=(4, 3), R=4)
img_R2K4x3 = get_Img(inpo_ksp)

plt.subplot(121)
plt.imshow(np.abs(img_R2K4x3), cmap='gray')
plt.title('GRAPPA, Ry=2, kernel4×3')
# plt.subplot(122)
# plt.imshow(np.abs(img_least_square_noiseCorrelation)-np.abs(img_R2K4x3), cmap='gray')
# plt.title('reconstruction error')
# plt.show()


RMSE = np.sqrt(np.sum((np.abs(img_least_square_noiseCorrelation) - np.abs(img_R2K4x3)) ** 2) / np.size(img_R2K4x3))
print("Root Mean Square Error is %1.5f (R=2)" % (RMSE))
