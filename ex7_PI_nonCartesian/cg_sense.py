import numpy
import torch
import torchkbnufft as tkbn
import numpy as np
import scipy
from typing import Optional

def cg_sense(data: np.ndarray,
             traj: np.ndarray,
             sens: np.ndarray,
             maxit: Optional[int] = 50,
             tol: Optional[float] = 1e-6,
             lambd: Optional[float] = 0,
             ) -> np.ndarray:

    # reconstruct subsampled PMRI data using CG SENSE [1]
    # uses M. Muckley's torchkbnufft: https://github.com/mmuckley/torchkbnufft
    # INPUT
    # data:    numpy 3D array [[kspace defined by traj] x coils] of coil data
    # traj:    trajectory, 2D array [read x lines] of complex trajectory points in [-0.5,0.5]
    # sens:    numpy 3D array of coil sensitivities [read x phase x coils]
    # lambd:   penalty for Tikhonov regularization
    # maxit:   maximum number of CG iterations
    # display: show iteration steps (1) or not (0)
    #
    # OUTPUT:
    #          reconstructed image
    #
    # Last Change: 03.01.2022
    # By: Bruno Riemenschneider
    #
    # [1] Pruessmann, K. P.; Weiger, M.; Boernert, P. and Boesiger, P.
    # Advances in sensitivity encoding with arbitrary k-space trajectories.
    # Magn Reson Med 46: 638-651 (2001)
    #
    # =========================================================================

    kdata = torch.permute(torch.flatten(torch.tensor(data), end_dim=1), [1, 0]).unsqueeze(0)
    smaps = torch.permute(torch.tensor(sens), [2, 0, 1]).unsqueeze(0)
    traj_x = np.real(traj) * 2 * np.pi
    traj_y = np.imag(traj) * 2 * np.pi
    traj_stack = torch.tensor(np.stack((traj_x.flatten(), traj_y.flatten()), axis=0))

    im_size = sens.shape[:-1]
    nufft_ob = tkbn.KbNufft(im_size=im_size)
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size)

    x = torch.zeros(sum(((1,1),im_size),()))
    r = adjnufft_ob(kdata, traj_stack, smaps=smaps, norm='ortho')
    p = r
    rs = torch.real(my_dot(r, r))
    gradient_norm_list = []

    for ii in range(maxit):
        Ap = adjnufft_ob(nufft_ob(p, traj_stack, smaps=smaps, norm='ortho'), traj_stack, smaps=smaps, norm='ortho') + lambd * p
        alpha = rs / torch.real(my_dot(p, Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        rs_next = torch.real(my_dot(r, r))
        if torch.sqrt(rs_next) < tol:
            break
        p = r + (rs_next / rs) * p
        rs = rs_next

        gradient_norm_list.append(np.sqrt(rs.numpy()))
        print('Iteration ' + str(ii+1) + ': residual norm is ' + str(np.sqrt(rs.numpy())))
        # residual does not take regularization into account

    return np.squeeze(x.detach().numpy()), gradient_norm_list

def my_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.vdot(torch.flatten(a),torch.flatten(b))