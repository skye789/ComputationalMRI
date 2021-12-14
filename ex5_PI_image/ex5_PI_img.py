import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import fractional_matrix_power  #scipy.linalg支持分数矩阵幂：
from sklearn.metrics import mean_squared_error

def ifft2c(kspace):
    kspace = np.fft.ifftshift(kspace)
    image = np.fft.ifft2(kspace)
    image = np.fft.fftshift(image)
    return image


# load matlab file
mat = scipy.io.loadmat('data_brain_8coils.mat')
c_coil_sen = mat['c']  #  coil sensitivity maps (256×256×8)
m_coil_ksp = mat['d']  # fully-sampled k-space 256×256×8  [PE,FE,channels],
noise = mat['n']  # noise-only scan (256×8)

nc = np.shape(m_coil_ksp)[2] #number of channel
sz_img = np.size(m_coil_ksp[...,0])

# for i in range(nc):
#     plt.subplot(1,nc,i+1)
#     plt.imshow(np.abs(ifft2c(m_coil_ksp[:,:,i])), cmap='gray')
#     plt.axis('off')
#     if i==nc/2: plt.title('8 coil images' )
# plt.show()

#######################################################
###### 1. Multicoil combination  ######
#######################################################

noise_cov = np.cov(np.transpose(noise.conjugate())) #8×8

#complex sum
img_complexSum = ifft2c(np.sum(m_coil_ksp,axis=2))

#sum of square
m_coil_img = np.zeros_like(m_coil_ksp)
for i in range(nc):
    m_coil_img[...,i] = ifft2c(m_coil_ksp[...,i])
img_sumSquare = np.linalg.norm(m_coil_img,axis=2)

#least square(match filter)
def least_square(m_coil_img,c_coil_sen):
    m_star_coil_img = m_coil_img.conjugate()
    f_img = np.zeros_like(c_coil_sen)
    for i in range(nc): # number of chanel
        f_img[...,i] = m_star_coil_img[...,i] * c_coil_sen[...,i]
    img_sum = np.sum(f_img,axis=2)      #256×256

    coil_sen2D = np.linalg.norm(c_coil_sen,axis=2)+10**-5 #256×256

    img_least_square = img_sum/coil_sen2D

    return img_least_square,coil_sen2D

img_least_square,coil_sen2D = least_square(m_coil_img,c_coil_sen)

brain_msk = np.copy(coil_sen2D)
brain_msk[coil_sen2D>10**-3] = 1
brain_msk[coil_sen2D<10**-3] = 0
# plt.imshow(brain_msk)
# plt.show()

# plt.subplot(131)
# plt.imshow(np.abs(img_complexSum), cmap='gray')
# plt.title('Complex sum')
# plt.axis('off')
# plt.subplot(132)
# plt.imshow(np.abs(img_sumSquare), cmap='gray')
# plt.title('Sum of squares')
# plt.axis('off')
# plt.subplot(133)
# plt.imshow(np.abs(img_least_square), cmap='gray')
# plt.title('Least square(matched filter)')
# plt.axis('off')
# plt.show()


#least square with_noiseCorrelation (match filter)
def least_square_with_noiseCorrelation(m_coil_img,c_coil_sen, noise_cov):
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

    #correct coil img and coil sensitivity with half inverse of noise covariance matrix
    m_w_coil_img = np.zeros_like(m_coil_img)
    c_w_coil_sen = np.zeros_like(c_coil_sen)
    for i in range(nc): #number of chanel
        for j in range(nc):
            m_w_coil_img[:,:,i] += n_cov_inv_half[i,j] * m_coil_img[:,:,j]
            c_w_coil_sen[:,:,i] += n_cov_inv_half[i,j] * c_coil_sen[:,:,j]

    m_w_star_coil_img = m_w_coil_img.conjugate()
    f_img = np.zeros_like(m_w_star_coil_img)
    for i in range(nc):
        f_img[...,i] = m_w_star_coil_img[...,i] * c_w_coil_sen[...,i]

    img_least_square = np.sum(f_img,axis=2)      #256×256
    coil_sen2D = np.linalg.norm(c_w_coil_sen,axis=2)+10**-5 #256×256
    img_least_square = img_least_square/coil_sen2D

    return img_least_square,coil_sen2D

img_least_square_noiseCorrelation,coil_sen2D_noiseCorrelation = least_square_with_noiseCorrelation(m_coil_img,c_coil_sen,noise_cov)
diff = np.abs(img_least_square)-np.abs(img_least_square_noiseCorrelation)

# plt.subplot(131)
# plt.imshow(np.abs(img_least_square), cmap='gray')
# plt.title('without coil noise covariance matrix')
# # plt.axis('off')
# plt.subplot(132)
# plt.imshow(np.abs(img_least_square_noiseCorrelation), cmap='gray')
# plt.title('with coil noise covariance matrix')
# plt.axis('off')
# plt.subplot(133)
# plt.imshow(diff, cmap='gray')
# plt.title('difference')
# plt.axis('off')
# plt.show()



#############################################################
###### 2. Cartesian SENSE reconstruction and g-factor  ######
#############################################################
def get_alias_img(m_coil_ksp, R):
    '''''
    Input:    
        m_coil_ksp: [Nx,Ny,Nc]
        R: acceleration factor
    Return:
        m_alias_img: aliase image [Nx,Ny/R,Nc] 

    '''''
    nx,ny,nc = np.shape(m_coil_ksp)

    m_alias_img = np.zeros((nx,int(np.ceil(ny/R)),nc),dtype=complex)
    for i in range(nc):
        m_alias_img[:,:,i] = ifft2c(m_coil_ksp[:,::R,i])

    return m_alias_img

# m_alias_img_2 = get_alias_img(m_coil_ksp, R=2)
# m_alias_img_3 = get_alias_img(m_coil_ksp, R=4)
# m_alias_img_4 = get_alias_img(m_coil_ksp, R=4)

# for i in range(nc):
#     plt.subplot(1,nc,i+1)
#     plt.imshow(np.abs(m_alias_img_4[:,:,i]), cmap='gray')
#     plt.axis('off')
#     if i==nc/2: plt.title('8 coil alias images, R=4' )
# plt.show()



def sense1d(m_alias_img, c_coil_sen, noise_cov, R):
    '''''
    Input:    
        m_alias_img: [Nx,Ny/R,Nc]
        c_coil_sen: [Nx,Ny,Nc]
        noise_cov: [Nc,Nc]
        R: acceleration factor
    Return:
        img_sense: unaliased image [Nx,Ny]
        g: g-factor map [Nx,Ny]
    '''''
    global g_val
    nx, ny, nc = np.shape(m_coil_ksp)

    # correct coil img and coil sensitivity with noise covariance matrix
    n_cov_inv_half = fractional_matrix_power(noise_cov, -1 / 2)
    m_w_alias_img = np.zeros_like(m_alias_img) #256×128×8
    c_w_coil_sen = np.zeros_like(c_coil_sen) #256×256×8
    for i in range(nc):  # number of chanel
        for j in range(nc):
            m_w_alias_img[:, :, i] += n_cov_inv_half[i, j] * m_alias_img[:, :, j]
            c_w_coil_sen[:, :, i] += n_cov_inv_half[i, j] * c_coil_sen[:, :, j]

    img_sen = np.zeros((nx, ny), dtype=complex)
    I = np.zeros((nc,1), dtype=complex)
    C = np.zeros((nc,R), dtype=complex)
    u = np.zeros((R,1), dtype=complex)
    g = np.zeros((nx, ny), dtype=complex)
    s_ny = int(np.ceil(ny/R))  #short ny
    for x in range(nx):
        for y in range(s_ny):
            for i in range(nc):
                I[i, 0] = m_w_alias_img[x, y, i]

                # create C for each pixel in alias image
                for r in range(R):
                    y_ = y + int((ny-s_ny)/2) + r * s_ny
                    if y_ < ny:
                        C[i, r] = c_w_coil_sen[x, y_, i]
                    else:
                        C[i, r] = c_w_coil_sen[x, y_-ny, i]

                #calculate 
                tmp0 = np.dot(np.transpose(C.conjugate()), C)
                if np.linalg.det(tmp0)==0:
                    m = 10**-6
                    tmp1 = np.linalg.inv(tmp0 + np.eye(R)*m)
                else:
                    tmp1 = np.linalg.inv(tmp0)
                tmp2 = np.dot(tmp1, np.transpose(C.conjugate()))
                u = np.dot(tmp2, I)
                g_val = np.sqrt(np.dot(np.diagonal(tmp1), np.diagonal(tmp0)))

            for r in range(R):
                y_ = y + int((ny-s_ny)/2) + r * s_ny
                if y_ < ny:
                    img_sen[x, y_] = u[r]
                    g[x, y_] = g_val
                else:
                    img_sen[x, y_ - ny] = u[r]
                    g[x, y_- ny] = g_val

    return img_sen, g



R = [2,3,4]
# img_index = [241, 242, 243,244,245, 246, 247, 248]
img_index = np.arange(331,340)
for i,r in enumerate(R):

        m_alias_img = get_alias_img(m_coil_ksp, R=r)
        img_sen,g_factor_map = sense1d(m_alias_img, c_coil_sen, noise_cov, R=r)

        plt.subplot(img_index[i])
        plt.imshow(np.abs(img_sen), cmap='gray')
        plt.title('SENSE'+str(r))
        plt.axis('off')

        plt.subplot(img_index[i+3])
        plt.imshow(np.abs(g_factor_map), cmap='gray')
        plt.title('g factor map R=' + str(r))
        plt.axis('off')

        recon_error = np.abs(img_least_square_noiseCorrelation) - np.abs(img_sen)
        plt.subplot(img_index[i + 6])
        plt.imshow(np.abs(recon_error), cmap='gray')
        plt.title('reconstruction error R=' + str(r))
        plt.axis('off')

        print("Accelatation factor R is %1.0f" %r)
        # Average g-factor
        g_mean = np.sum(np.abs(g_factor_map)*brain_msk)/np.sum(brain_msk)
        print("Average g-factor is %1.2f " %(g_mean))

        # SNR
        S = np.sum(np.abs(img_least_square_noiseCorrelation)*brain_msk)/np.sum(brain_msk) #average pixel signal amplitude in brain region
        N = np.std(np.abs(img_least_square_noiseCorrelation)[45:55,45:55])
        SNR = S/N
        print("SNR(non-acc) is %1.5f " % (SNR))

        S_sense = np.sum(np.abs(img_sen)*brain_msk)/np.sum(brain_msk) #average pixel signal amplitude in brain region
        N_sense = np.std(np.abs(img_sen)[45:55,45:55])
        SNR_sense = S_sense/N_sense
        print("SNR(acc) is %1.5f " % (SNR_sense))

        # RMSE = mean_squared_error(np.abs(img_least_square_noiseCorrelation).flatten(),  np.abs(img_sen).flatten(), squared=False)
        RMSE = np.sqrt(np.sum((np.abs(img_least_square_noiseCorrelation)-np.abs(img_sen))**2)/np.size(img_sen))
        print("Root Mean Square Error is %1.5f " %(RMSE))
        print()

plt.show()


