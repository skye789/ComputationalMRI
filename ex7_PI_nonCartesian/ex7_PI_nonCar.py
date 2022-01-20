import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchkbnufft as tkbn
from matplotlib.ticker import MaxNLocator
from typing import Optional
from cg_sense import cg_sense
from grid import grid

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def fft2c(img): #img原点在左上
    kspace = np.fft.fft2(img)
    kspace = np.fft.fftshift(kspace)
    return kspace

def ifft2c(kspace):
    image = np.fft.ifft2(kspace)
    img = np.fft.ifftshift(image)
    return img


def NUFFT(radial_kdata, coil_sen, w, sz_img):
    '''''
    Input:
        radial_kdata:[512,64,4] 
        w: density compensation [512,64]
        sz_img: size of image
    Output:
        car_img:[256,256]
    '''''
    car_img = np.zeros((sz_img, sz_img), dtype=complex)
    coil_img = np.zeros_like(coil_sen, dtype=complex)
    for i in range(num_channel):
        compensated_ratial_kdata = radial_kdata[:,:,i] * w  #density compensation
        kspace_catesian_grid = grid(compensated_ratial_kdata, radial_traj, sz_img)
        coil_img[:, :, i] = ifft2c(kspace_catesian_grid)
        car_img += coil_img[:, :, i] * coil_sen[:, :, i].conjugate()
    return car_img

    # INPUT
    # data:    numpy 3D array [[kspace defined by traj] x coils] of coil data
    # traj:    trajectory, 2D array [read x lines] of complex trajectory points in [-0.5,0.5]
    # sens:    numpy 3D array of coil sensitivities [read x phase x coils]
    # maxit:   maximum number of CG iterations
    # display: show iteration steps (1) or not (0)
    #
    # OUTPUT:
    #          reconstructed image

def gradientDescent(data: np.ndarray,
             traj: np.ndarray,
             sens: np.ndarray,  #coil sensitivity
             maxit: Optional[int],
             t=0.01
             ):
    kdata = torch.permute(torch.flatten(torch.tensor(data), end_dim=1), [1, 0]).unsqueeze(0)
    smaps = torch.permute(torch.tensor(sens), [2, 0, 1]).unsqueeze(0)
    traj_x = np.real(traj) * 2 * np.pi
    traj_y = np.imag(traj) * 2 * np.pi
    traj_stack = torch.tensor(np.stack((traj_x.flatten(), traj_y.flatten()), axis=0))
    im_size = sens.shape[:-1]

    nufft_ob = tkbn.KbNufft(im_size=im_size)    # from reconstructied img to ratial trajectory
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size)  # from ratial trajectory to reconstructied img

    u = torch.zeros(sum(((1, 1), im_size), ()), dtype=torch.cdouble)  #img
    g = kdata
    mse_list= []
    gradient_norm_list = []
    for ii in range(maxit):
        Ku = nufft_ob(u, traj_stack ,smaps=smaps, norm='ortho')
        KT = adjnufft_ob(Ku-g, traj_stack, smaps=smaps, norm='ortho')
        u = u - t * 2 * KT
        gradient_norm = scipy.linalg.norm(2 * KT)
        img = np.squeeze(u.detach().numpy())

        mse = np.sum((np.abs(img)-np.abs(img_senscomb))**2)/np.size(img)
        gradient_norm_list.append(gradient_norm)
        mse_list.append(mse)
    return np.squeeze(x.detach().numpy()), mse_list, gradient_norm_list


if __name__=="__main__":
    # load matlab file
    mat = scipy.io.loadmat('data_radial_brain_4ch.mat')
    radial_kdata = mat['kdata']  # [512,64,4]  [readout points,nspokes,channels]
    coil_sen = mat['c']  # coil_sen[256,256,4]
    radial_traj = mat['k']  # [512,64]
    w = mat['w']  # density compensation [512,64]
    img_senscomb = mat['img_senscomb'] # [256, 256]  Sensitivity combined fully sampled ground truth
    readout_points, nspokes, num_channel = np.shape(radial_kdata)
    sz_img = len(coil_sen)

    NUFFT_img = NUFFT(radial_kdata, coil_sen, w, sz_img)

    ###################################################################
    ###### 2. Iterative image reconstruction with gradient descent  ###
    ###################################################################
    iter_num = 5
    type = 'cgnoise' # gd cg cgnoise
    if type== 'gd':
        img, mse_list, gradient_norm_list = gradientDescent(radial_kdata,radial_traj,coil_sen, iter_num)

        plt.subplot(221)
        plt.title("Gradient descent, iter:"+ str(iter_num) +', t=0.01')
        plt.imshow(np.abs(img),cmap='gray')
        plt.subplot(222)
        plt.title('Ground Truth')
        plt.imshow(np.abs(img_senscomb),cmap='gray')
        plt.subplot(223)
        plt.title('Difference')
        plt.imshow(np.abs(img)-np.abs(img_senscomb),cmap='gray')
        plt.subplot(224)
        plt.title('NUFFT')
        plt.imshow(np.abs(NUFFT_img),cmap='gray')
        # plt.savefig('./result/gradientDescent.png')
        plt.show()

        x = np.arange(iter_num)+1
        plt.subplot(121)
        plt.plot(x, gradient_norm_list)
        plt.title('L2 norm of gradient(Gradient Descent), Iteration number='+str(iter_num))
        plt.subplot(122)
        plt.plot(x, mse_list)
        plt.title('Mean Square Error')

        plt.savefig('./result/mse&gradientNorm.png')
        plt.show()


    #########################################################
    ###### 3. CG-SENSE  #####################################
    #########################################################
    if type== 'cg':
        img,gradient_norm_list = cg_sense(radial_kdata, radial_traj, coil_sen, iter_num)

        plt.subplot(221)
        plt.title("Conjugate Gradient, iter:"+ str(iter_num))
        plt.imshow(np.abs(img),cmap='gray')
        plt.subplot(222)
        plt.title('Ground Truth')
        plt.imshow(np.abs(img_senscomb),cmap='gray')
        plt.subplot(223)
        plt.title('Difference')
        plt.imshow(np.abs(img)-np.abs(img_senscomb),cmap='gray')
        plt.subplot(224)
        plt.title('NUFFT')
        plt.imshow(np.abs(NUFFT_img),cmap='gray')

        plt.savefig('./result/Conjugate Gradient.png')
        plt.show()

        x = np.arange(iter_num) + 1
        plt.plot(x, gradient_norm_list)
        plt.title('L2 norm of gradient(Conjugate Gradient), Iteration number='+str(iter_num))

        plt.savefig('./result/mse&gradientNorm(CG).png')
        plt.show()

    if type=='cgnoise':
        noise = np.random.normal(0, 0.1, radial_kdata.shape).astype(complex)
        img,_ = cg_sense(noise, radial_traj, coil_sen, maxit=500)
        print(img.shape)
        noise_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))


        plt.title("Conjugate Gradient, noise in kspace, iter:500" )
        plt.imshow(np.log(np.abs(noise_fft)), cmap='gray')
        plt.savefig('./result/noise(CG method) in kspace.png')
        plt.show()

