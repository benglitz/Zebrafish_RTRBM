import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import pearsonr
from itertools import permutations
import matplotlib.pyplot as plt
import sklearn

def reshape_from_batches(x, mode='stack_batches'):
    if mode == 'stack_batches':
        return torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
    elif mode == 'stack_visibles':
        return torch.reshape(x, (x.shape[0] * x.shape[2], x.shape[1]))


def reshape_to_batches(spikes, mini_batch_size=128):
    # reshape to train in batches
    mini_batch_size = mini_batch_size
    nr_batches = (spikes.shape[1] // mini_batch_size)
    spikes = spikes[:, :mini_batch_size * nr_batches]
    V = torch.zeros([spikes.shape[0], mini_batch_size, nr_batches])
    for j in range(nr_batches):
        V[:, :, j] = spikes[:, mini_batch_size * j:mini_batch_size * (j + 1)]

def get_pw_shifted(data, k=1):
    return torch.matmul(data[:, :-k], data[:, k:].T) / (data.shape[1] - k)


def get_reconstruction_mean_pairwise_correlations(true_data, sampled_data, n=1000, m=50000):

    # reshape data if it is still in batches
    if true_data.dim() == 3:
        true_data = reshape_from_batches(true_data)
    if sampled_data.dim() == 3:
        sampled_data = reshape_from_batches(sampled_data)

    # reduce shape to reduce computation time
    if true_data.shape[0] > n:
        idx = torch.randperm(true_data.shape[0])[:n]
        true_data = true_data[idx, :]
        sampled_data = sampled_data[idx, :]

    # get correlation of true and reconstructed data
    reconstruction_correlation, _ = pearsonr(true_data.flatten(), sampled_data.flatten())

    # calculate first order moments
    true_moments, sampled_moments = torch.mean(true_data, 1), torch.mean(sampled_data, 1)

    # get correlation of true and reconstructed first order moments
    mean_correlation, _ = pearsonr(true_moments, sampled_moments)

    # calculate second order moments
    true_pairwise = pairwise_moments(true_data, true_data).flatten()
    sampled_pairwise = pairwise_moments(sampled_data, sampled_data).flatten()

    # reduce shape to reduce computation time
    if true_pairwise.shape[0] > m:
        idx = torch.randperm(true_pairwise.shape[0])[:m]
        true_pairwise = true_pairwise[idx]
        sampled_pairwise = sampled_pairwise[idx]

    # get correlation of true and reconstructed second order moments
    pairwise_correlation, _ = pearsonr(true_pairwise, sampled_pairwise)

    # return
    return reconstruction_correlation, mean_correlation, pairwise_correlation


def calculate_correlation(x, y):
    return pearsonr(x, y)


def set_to_device(rtrbm, device):
    rtrbm.errors = torch.tensor(rtrbm.errors, device=device)
    rtrbm.W = rtrbm.W.to(device).detach().clone()
    rtrbm.U = rtrbm.U.to(device).detach().clone()
    rtrbm.b_V = rtrbm.b_V.to(device).detach().clone()
    rtrbm.b_H = rtrbm.b_H.to(device).detach().clone()
    rtrbm.b_init = rtrbm.b_init.to(device).detach().clone()
    rtrbm.V = rtrbm.V.to(device).detach().clone()
    rtrbm.device = device

    return


def pairwise_moments(data1, data2):
    """Average matrix product."""
    return torch.matmul(data1, data2.T) / torch.numel(data1)


def RMSE(test, est):
    """Calculates the Root Mean Square Error of two vectors."""

    return torch.sqrt(torch.sum((test - est) ** 2) / torch.numel(test))


def nRMSE(train, test, est):
    """Calculates the normalised Root Mean Square Error of two statistics vectors, given training data statistics."""

    rmse = RMSE(test, est)

    test_shuffled = test[torch.randperm(int(test.shape[0]))]
    est_shuffled = est[torch.randperm(int(est.shape[0]))]

    rmse_shuffled = RMSE(test_shuffled, est_shuffled)
    rmse_optimal = RMSE(train, test)

    return 1 - (rmse - rmse_shuffled) / (rmse_optimal - rmse_shuffled)


def free_energy(v, W, b_V, b_H):
    """Get free energy of RBM"""
    v_term = torch.outer(v, b_V.T)
    w_x_h = torch.nn.functional.linear(v, W.T, b_H)
    h_term = torch.sum(torch.nn.functional.softplus(w_x_h))
    free_energy = torch.mean(-h_term - v_term)

    return free_energy


def get_nRMSE_moments(model, V_train, V_test, V_est, H_train, H_test, H_est, sp=0):
    """ Calculates normalised Root Mean Square Error of moments and pairwise moments """

    # <v_i>
    V_mean_train = torch.mean(V_train, 1)
    V_mean_test = torch.mean(V_test, 1)
    V_mean_est = torch.mean(V_est, 1)

    # <h_{mu}>
    H_mean_train = torch.mean(H_train, 1)
    H_mean_test = torch.mean(H_test, 1)
    H_mean_est = torch.mean(H_est, 1)

    # <v_i h_{mu}>_{model} = <v_i h_{mu}>_{model-generated data} + lamda*sign(w_{i,mu})
    VH_mgd_train = pairwise_moments(V_train, H_train)
    VH_mgd_test = pairwise_moments(V_test, H_test)
    VH_mgd_est = pairwise_moments(V_est, H_est)

    VH_mean_train = VH_mgd_train + sp * torch.sign(model.W.T)
    VH_mean_test = VH_mgd_test + sp * torch.sign(model.W.T)
    VH_mean_est = VH_mgd_est + sp * torch.sign(model.W.T)

    # <v_i v_j> - <v_i><v_j>
    VV_mean_train = pairwise_moments(V_train, V_train) - torch.outer(V_mean_train, V_mean_train)
    VV_mean_test = pairwise_moments(V_test, V_test) - torch.outer(V_mean_test, V_mean_test)
    VV_mean_est = pairwise_moments(V_est, V_est) - torch.outer(V_mean_est, V_mean_est)

    # <h_i h_j> - <h_i><h_j>
    HH_mean_train = pairwise_moments(H_train, H_train) - torch.outer(H_mean_train, H_mean_train)
    HH_mean_test = pairwise_moments(H_test, H_test) - torch.outer(H_mean_test, H_mean_test)
    HH_mean_est = pairwise_moments(H_est, H_est) - torch.outer(H_mean_est, H_mean_est)

    V_nRMSE = nRMSE(V_mean_train, V_mean_test, V_mean_est)
    H_nRMSE = nRMSE(H_mean_train, H_mean_test, H_mean_est)
    VH_nRMSE = nRMSE(VH_mean_train, VH_mean_test, VH_mean_est)
    VV_nRMSE = nRMSE(VV_mean_train, VV_mean_test, VV_mean_est)
    HH_nRMSE = nRMSE(HH_mean_train, HH_mean_test, HH_mean_est)

    return V_nRMSE, H_nRMSE, VH_nRMSE, VV_nRMSE, HH_nRMSE


def correlation(v):
    return np.corrcoef(v)


# def mutual_information(v_prob):
#    for t in range(v_prob.shape[1]-1):
#        MU[:,:,t] = torch.outer(v_prob[:,t], v_prob[:,t+1]) * torch.log()
#    return 9

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


def correlation_matrix(data):
    # data.shape = [n, T]f
    population_vector = np.array(data)
    C = np.zeros((population_vector.shape[0], population_vector.shape[0]))
    for i in range(population_vector.shape[0]):
        for j in range(population_vector.shape[0]):
            C[i][j] = pearsonr(population_vector[i], population_vector[j])[0]
    return C


def cross_correlation(data, time_shift=1, mode='Correlate'):
    data = np.array(data)
    time_shift = int(time_shift)
    if data.ndim==3:
        for s in range(data.shape[2]):
            if time_shift == 0:
                population_vector_t = np.array(data)
                population_vector_tm = np.array(data)
            elif time_shift != 0:
                population_vector_t = np.array(data[:, time_shift:, s])
                population_vector_tm = np.array(data[:, :-time_shift, s])
            C = np.zeros([population_vector_t.shape[0], population_vector_tm.shape[0], data.shape[2]])
            for i in range(population_vector_t.shape[0]):
                for j in range(population_vector_tm.shape[0]):
                    if mode == 'Correlate':
                        C[i][j][s] = np.correlate(population_vector_t[i], population_vector_tm[j])
                    elif mode == 'Pearson':
                        C[i][j][s] = np.corrcoef(population_vector_t[i], population_vector_tm[j])[1, 0]
        C = np.mean(C, 2)

    elif data.ndim==2:
        if time_shift == 0:
            population_vector_t = np.array(data)
            population_vector_tm = np.array(data)
        elif time_shift != 0:
            population_vector_t = np.array(data[:, time_shift:])
            population_vector_tm = np.array(data[:, :-time_shift])

        C = np.zeros([population_vector_t.shape[0], population_vector_tm.shape[0]])
        for i in range(population_vector_t.shape[0]):
            for j in range(population_vector_tm.shape[0]):
                if mode == 'Correlate':
                    C[i][j] = np.correlate(population_vector_t[i], population_vector_tm[j])
                elif mode == 'Pearson':
                    C[i][j] = np.corrcoef(population_vector_t[i], population_vector_tm[j])[1, 0]
    return torch.tensor(C)

def create_U_hat(n_h):
    U_hat = torch.zeros(n_h, n_h)
    U_hat += torch.diag(torch.ones(n_h - 1), diagonal=-1)
    U_hat += torch.diag(-torch.ones(n_h - 1), diagonal=1)
    U_hat[0, -1] = 1
    U_hat[-1, 0] = -1
    return U_hat


def shuffle_back(W_trained, U_trained, W_true=None, U_true=None):
    if U_true is not None:
        n_h, n_v = W_trained.shape
        corr = np.zeros((n_h, n_h))
        shuffle_idx = np.zeros((n_h))
        for i in range(n_h):
            for j in range(n_h):
                corr[i, j] = np.correlate(U_trained[j, :], U_true[i, :])
            shuffle_idx[i] = np.argmax(corr[i, :])
        W_trained = W_trained[shuffle_idx, :]
        U_trained = U_trained[shuffle_idx, :]
        U_trained = U_trained[:, shuffle_idx]

    if W_true is not None:
        n_h, n_v = W_trained.shape
        corr = np.zeros([n_h, n_h])
        shuffle_idx = np.zeros(n_h)
        for i in range(n_h):
            for j in range(n_h):
                corr[i, j] = np.correlate(W_true[i, :], np.abs(W_trained[j, :]))
            shuffle_idx[i] = np.argmax(corr[i, :])
        W_trained = W_trained[shuffle_idx, :]
        if U_trained is None:
            return W_trained
        else:
            U_trained = U_trained[shuffle_idx, :]
            U_trained = U_trained[:, shuffle_idx]
    return W_trained, U_trained


def get_best_correlation(U, U_hat, mode='max'):
    # number of hidden units
    n_h = int(U.shape[0])

    # create empty np array to save correlations
    corrs = np.empty(np.math.factorial(n_h))

    # get all possible permutations of hidden units
    perms = permutations(range(0, n_h))

    # loop over all permutations
    for i, idx in enumerate(perms):

        # permute U
        U_ = U[idx, :]
        U_ = U_[:, idx]

        # calculate correlation with true weights
        correlation = np.corrcoef(U_.flatten(), U_hat.flatten())[0, 1]

        # save correlation
        corrs[i] = correlation

    # return max value
    if mode == 'max':
        return np.max(corrs)

    # return mean value
    elif mode == 'mean':
        return np.mean(corrs)

    else:
        raise ValueError('"mode" must be "max" or "mean"')

def reshape(data, T=None, n_batches=None, dtype=torch.float):
    data = torch.tensor(data)
    if n_batches == None:
        if data.ndim == 2:
            raise ValueError('Already in right shape')
        N, T, num_samples = data.shape
        data1 = torch.zeros(N, T * num_samples)
        for i in range(num_samples):
            data1[:, T * i:T * (i + 1)] = data[:, :, i]

    elif n_batches and T is not None:
        N, _ = data.shape
        data1 = torch.zeros(N, T, n_batches)
        for i in range(n_batches):
            data1[:, :, i] = data[:, T * i:T * (i + 1)]
    else:
        raise ValueError('Specify n_batches and T')

    return data1


def resample(data, sr, mode=2):
    '''
    :param data: original data
    :param sr: sampling rate
    :param mode: =1 take the mean, =2 take instance value
    :return: downsampled data
    '''

    if data.ndim == 3:
        N_V, T, n_batches = data.shape
        new_data = np.array(data)

        # make sure that the modulus(T/sr) = 0
        if T % sr != 0:
            new_data = new_data[:, :int(np.floor(T / sr)), :]
        s = int(np.floor(T / sr))
        data_nsr = np.zeros([N_V, s, n_batches])

        for batch in range(int(n_batches / 20)):
            for t in range(s):
                if mode == 1:
                    temp_data = np.mean(new_data[:, sr * t:sr * (t + 1), 20 * batch:20 * (batch + 1)], 1)
                    temp_data.ravel()[temp_data.ravel() > 0.5] = 1.0
                    temp_data.ravel()[temp_data.ravel() <= 0.5] = 0.0
                    # temp_data.ravel()[temp_data.ravel() == 0.5] = 1.0 * (np.random.rand(np.sum(temp_data == 0.5)) > 0.5)
                    data_nsr[:, t, 20 * batch:20 * (batch + 1)] = temp_data

                elif mode == 2:
                    data_nsr[:, t, 20 * batch:20 * (batch + 1)] = data[:, sr * t, 20 * batch:20 * (batch + 1)]

    elif data.ndim == 2:
        N_V, T = data.shape
        new_data = np.array(data)

        # make sure that the modulus(T/sr) = 0
        if T % sr != 0:
            new_data = new_data[:, :int(np.floor(T / sr))]
        s = int(np.floor(T / sr))
        data_nsr = np.zeros([N_V, s])

        for t in range(s):
            if mode == 1:
                temp_data = np.mean(new_data[:, sr * t:sr * (t + 1)], 1)
                temp_data.ravel()[temp_data.ravel() > 0.5] = 1.0
                temp_data.ravel()[temp_data.ravel() <= 0.5] = 0.0
                # temp_data.ravel()[temp_data.ravel() == 0.5] = 1.0 * (np.random.rand(np.sum(temp_data == 0.5)) > 0.5)
                data_nsr[:, t] = temp_data

            elif mode == 2:
                data_nsr[:, t] = data[:, sr * t]

    return torch.tensor(data_nsr, dtype=torch.float)

def train_test_split(data, train_batches=80, test_batches=20):
    n_batches = train_batches + test_batches
    batch_size = data.shape[1] // n_batches
    train = torch.zeros(data.shape[0], batch_size, train_batches)
    test = torch.zeros(data.shape[0], batch_size, test_batches)

    batch_index_shuffled = torch.randperm(n_batches)
    i = 0
    for batch in range(train_batches):
        j = batch_index_shuffled[i]
        train[:, :, batch] = data[:, j * batch_size:(j + 1) * batch_size]
        i += 1

    for batch in range(test_batches):
        j = batch_index_shuffled[i]
        test[:, :, batch] = data[:, j * batch_size:(j + 1) * batch_size]
        i += 1

    return train, test


def optimal_cluster_plot(X, n_clusters=10):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    from scipy.stats import pearsonr
    # Optimal cluster according to silhouette and elbow method
    plt.figure(figsize=(10, 5))

    inertia = []
    for i in range(1, n_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++')
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_clusters + 1), inertia)
    plt.title('Elbow Method', fontsize=20)
    plt.xlabel('Number of clusters', fontsize=20)
    plt.ylabel('Inertia', fontsize=20)

    s_score = []
    for i in range(2, n_clusters + 1):  # note that we start from 2: the silhouette is not defined for a single cluster
        kmeans = KMeans(n_clusters=i, init='k-means++')
        labels = kmeans.fit_predict(X)
        s_score.append(silhouette_score(X, labels))

    plt.subplot(1, 2, 2)
    plt.plot(range(2, n_clusters + 1), s_score)
    plt.title('Silhouette', fontsize=20)
    plt.xlabel('Number of clusters', fontsize=20)
    plt.ylabel('silhouette score', fontsize=20)

    plt.tight_layout()
    plt.show()