import torch
import numpy as np
from tqdm import tqdm


def get_param_history(parameter_history):
    """
    Returns parameter history per parameter as torch tensor
    """
    epochs = len(parameter_history)
    N_H, N_V = np.shape(parameter_history[0][0])
    W = torch.empty(epochs, N_H, N_V)
    U = torch.empty(epochs, N_H, N_H)
    b_V = torch.empty(epochs, 1, N_V)
    b_H = torch.empty(epochs, 1, N_H)
    b_init = torch.empty(epochs, 1, N_H)

    for ep, params in enumerate(parameter_history):
        W[ep] = params[0].clone().detach()
        U[ep] = params[1].clone().detach()
        b_H[ep] = params[2].clone().detach()
        b_V[ep] = params[3].clone().detach()
        b_init[ep] = params[4].clone().detach()

    return W, U, b_H, b_V, b_init


def set_to_device(rtrbm, device):
    rtrbm.errors = torch.tensor(rtrbm.errors, device=device)
    rtrbm.W = rtrbm.W.to(device).detach().clone()
    rtrbm.U = rtrbm.U.to(device).detach().clone()
    rtrbm.b_V = rtrbm.b_V.to(device).detach().clone()
    rtrbm.b_H = rtrbm.b_H.to(device).detach().clone()
    rtrbm.b_init = rtrbm.b_init.to(device).detach().clone()
    rtrbm.V = rtrbm.V.to(device).detach().clone()
    rtrbm.device = device


def make_voxel_xyz(n, spikes, xyz, mode=1, fraction=0.5, disable_tqdm=False):
    n = n + 1  # number of voxels
    x = torch.linspace(torch.min(xyz[:, 0]), torch.max(xyz[:, 0]), n)
    y = torch.linspace(torch.min(xyz[:, 1]), torch.max(xyz[:, 1]), n)
    z = torch.linspace(torch.min(xyz[:, 2]), torch.max(xyz[:, 2]), n)

    voxel_xyz = torch.zeros((n - 1) ** 3, 3)
    voxel_spike = torch.zeros((n - 1) ** 3, spikes.shape[1])
    i = 0
    for ix in tqdm(range(n - 1), disable=disable_tqdm):
        for iy in range(n - 1):
            for iz in range(n - 1):
                condition = ((xyz[:, 0] > x[ix]) & (xyz[:, 0] < x[ix + 1]) & (xyz[:, 1] > y[iy]) & \
                             (xyz[:, 1] < y[iy + 1]) & (xyz[:, 2] > z[iz]) & (xyz[:, 2] < z[iz + 1]))

                if torch.sum(condition) == 0:
                    continue
                V = spikes[condition, :]
                if mode == 1:
                    voxel_spike[i, :] = torch.mean(V, 0)
                if mode == 2:
                    voxel_spike[i, :] = torch.max(V, 0)[0]
                if mode == 3:
                    voxel_spike[i, :] = torch.mean(
                        torch.sort(V, dim=0, descending=True)[0][:int(np.ceil(fraction * V.shape[0])), :], 0)

                voxel_xyz[i, 0] = x[ix]
                voxel_xyz[i, 1] = y[iy]
                voxel_xyz[i, 2] = z[iz]
                i += 1

    condition = ((voxel_xyz[:, 0] > 0) & (voxel_xyz[:, 1] > 0) & (voxel_xyz[:, 2] > 0))
    voxel_xyz = voxel_xyz[condition, :]
    voxel_spike = voxel_spike[condition, :]

    return [voxel_spike, voxel_xyz]


def get_hidden_mean_receptive_fields(weights, coordinates, only_max_conn=False):
    """
        Computes the receptive fields of the hidden units.

        Parameters
        ----------
        VH : torch.Tensor
            The hidden layer's weight matrix.
        coordinates : torch.Tensor
            The coordinates of the visible units.
        only_max_conn : bool, optional
            If True, only the receptive field of the unit with the maximal
            connection to the hidden layer is returned.

        Returns
        -------
        torch.Tensor
            The receptive fields of the hidden units. """

    VH = weights.detach().clone()

    if only_max_conn is False: VH[VH < 0] = 0

    n_dimensions = torch.tensor(coordinates.shape).shape[0]
    N_H = VH.shape[0]

    max_hidden_connection = torch.max(VH, 0)[1]
    if n_dimensions == 1:
        rf = torch.zeros(N_H)
        for h in range(N_H):
            if only_max_conn:
                v_idx = (max_hidden_connection == h)
                rf[h] = torch.mean(coordinates[v_idx])
            else:
                rf[h] = torch.sum(VH[h, :] * coordinates / torch.sum(VH[h, :]))
    else:
        rf = torch.zeros(N_H, n_dimensions)
        for i in range(n_dimensions):
            for h in range(N_H):
                if only_max_conn:
                    v_idx = (max_hidden_connection == h)
                    rf[h, i] = torch.mean(coordinates[v_idx, i])
                else:
                    rf[h, i] = torch.sum(VH[h, :] * coordinates[:, i] / torch.sum(VH[h, :]))

    return rf