import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import fractional_matrix_power
from ex6_PI_kspace.grappaR2K2x3 import grappaR2K2x3


def ifft2c(kspace):
    image = np.fft.ifft2(kspace)
    img = np.fft.ifftshift(image)
    return img


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
    nx,_,_ = fs_kdata.shape
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
    nx, _, _ = fs_kdata.shape
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


class Grappa:
    '''''
    Input:
        zf_kspace: zero filled kspace every R-1 rows, 
        acs: auto calibration signal. Usually the central several lines of fully sampled kspace 
        R: accelaration factor
    Output:
        reconstructed kspace
    '''''
    def __init__(self, zf_kspace, acs, R):
        self.R = R
        self.zf_kspace = zf_kspace
        self.num_row_zfKspace, self.num_col_zfKspace,_ = zf_kspace.shape

        self.acs = acs
        self.num_row_acs, self.num_col_acs, self.nc = acs.shape

        self.kernel_shape = (4,3)
        self.num_row_kernel, self.num_col_kernel = self.kernel_shape  # 4×3

        self.block_width = self.num_col_kernel
        self.block_height = (self.num_row_kernel - 1) * self.R + 1

        self.nb = int((self.num_row_acs-(self.block_height-1)) * (self.num_col_acs - 2)) #10=(R+1)*2
        self.nk = self.num_row_kernel * self.num_col_kernel  # kernel size, 12

        self.row_offset = np.arange(R + 1, R * 2)  #row offset from top left of block to target


    def extract(self):
        '''''
        from acs extract source and target                
        '''''
        src = np.zeros((self.R-1,self.nb, self.nc * self.nk), dtype=np.complex64)
        targ = np.zeros((self.R-1, self.nb, self.nc), dtype=np.complex64)

        for i in range(self.R-1):
            src_idx = 0
            for col in range(self.num_col_acs - 2):
                for row in range( self.num_row_acs-(self.block_height-1)):  # for循环所有target点(nb)

                    block = self.acs[row:row+self.block_height, col:col+self.block_width]

                    src[i,src_idx] = block[::self.R].flatten()
                    targ[i,src_idx] = acs[row+self.row_offset[i], col+1]

                    src_idx += 1

        return src, targ



    def interp(self, zp_kspace, ws):
        interpolated = np.array(zp_kspace)
        num_row_zpk, num_col_zpk, nc = zp_kspace.shape

        for i in range(self.R-1):
            src_idx = 0
            for col in range(num_col_zpk - 2):
                for row in range(0, num_row_zpk-(self.block_height-1), self.R):  # for循环所有target点(nb)

                    block = zp_kspace[row:row+self.block_height, col:col+self.block_width]
                    src = block[::self.R].flatten()

                    interpolated[row+self.row_offset[i] , col+1] = np.dot(src, ws[i])

                    src_idx += 1

        return interpolated


    def zero_padding(self, zp_up, zp_down):

        zp_kdata = np.zeros((self.num_row_zfKspace + zp_up+zp_down, self.num_col_zfKspace + 2, nc), np.complex64)
        zp_kdata[zp_up:self.num_row_zfKspace+zp_up, 1:self.num_col_zfKspace + 1, :] = self.zf_kspace
        return zp_kdata

    def grappa4x3(self,flag_acs):
        # get source and target in acs region, to calculate weight
        ws = np.zeros((self.R-1,self.nk*self.nc, self.nc), dtype=complex)
        src, targ = self.extract()  # 3维度
        for i in range(self.R-1):
            ws[i] = np.dot(np.linalg.pinv(src[i]), targ[i])

        # zero padding
        zp_up = self.R
        zp_down = self.R + 1
        zp_kdata = self.zero_padding(zp_up, zp_down)

        # interpolation
        interpolated = self.interp(zp_kdata, ws)
        interpolated = interpolated[zp_up:self.num_row_zfKspace+zp_up, 1:self.num_col_zfKspace + 1, :]

        if flag_acs:
            interpolated[int(self.num_row_zfKspace / 2 - self.num_row_acs / 2)
                         :int(self.num_row_zfKspace / 2 + self.num_row_acs / 2)] = self.acs

        return interpolated


if __name__=="__main__":
    # load matlab file
    mat = scipy.io.loadmat('data_brain_8coils.mat')
    c_coil_sen = mat['c']  # coil sensitivity maps (256×256×8)
    fs_kdata = mat['d']  # fully-sampled k-space 256×256×8  [PE,FE,channels],
    noise = mat['n']  # noise-only scan (256×8)

    _,_, nc = np.shape(fs_kdata)
    m_coil_img = np.zeros_like(fs_kdata)
    for i in range(nc):
        m_coil_img[..., i] = ifft2c(fs_kdata[..., i])
    noise_cov = np.cov(np.transpose(noise.conjugate()))  # 8×8
    img_least_square, _ = least_square(m_coil_img,c_coil_sen )

    #############################################################
    ###### 1. Simple GRAPPA reconstruction  ######
    #############################################################
    zf_kspace = get_zf_kspace(R=2, fs_kdata=fs_kdata)
    acs = get_acs(num_cenLine=48, fs_kdata=fs_kdata)
    inpo_ksp, _ = grappaR2K2x3(zf_kspace, acs, flag_acs = False)
    img_R2K2x3 = get_Img(inpo_ksp)

    # plt.imshow(np.log(np.abs(zf_kspace[...,0])))
    # plt.imshow(np.abs(img_R2K2x3), cmap='gray')
    # plt.title('GRAPPA, Ry=2, kernel2×3')
    # plt.show()

    #############################################################
    ###### 2. Modify GRAPPA reconstruction  ######
    #############################################################
    subplot_list = np.arange(231,237)
    R_list = np.array([2,3,4])
    for i,R in enumerate(R_list):
        zf_kspace = get_zf_kspace(R, fs_kdata=fs_kdata)
        acs = get_acs(48, fs_kdata=fs_kdata)
        grappa = Grappa(zf_kspace,acs,R)
        inpo_ksp= grappa.grappa4x3(flag_acs=False)
        img_GRAPPA = get_Img(inpo_ksp)

        error_img = img_GRAPPA-img_least_square

        plt.subplot(subplot_list[i])
        plt.imshow(np.abs(img_GRAPPA), cmap='gray')
        plt.title('GRAPPA, Ry=' + str(R) +' kernel4×3')
        plt.axis('off')
        plt.subplot(subplot_list[i]+3)
        plt.imshow(np.abs(error_img), cmap='gray')
        plt.title('reconstruction error, Ry=' + str(R))
        plt.axis('off')
        print('sum of reconstruction error(R=%1.0f) is ' %(R),np.sum(np.abs(error_img)),)

        RMSE = np.sqrt(np.sum((np.abs(img_least_square)-np.abs(img_GRAPPA))**2)/np.size(img_GRAPPA))
        print("Root Mean Square Error(R=%1.0f) is %1.5f " %(R,RMSE))

    plt.show()




