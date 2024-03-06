import numpy as np
import torch
from tqdm import tqdm
from itertools import permutations
import sklearn
from scipy.stats import pearsonr, spearmanr, ranksums, kruskal, wilcoxon, ttest_ind, f_oneway

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterExponent
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

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


def shuffle_back(W_trained, U_trained, W_true, allow_duplicates=False):
    
    n_h, n_v = W_trained.shape
    corr = np.zeros([n_h, n_h])
    shuffle_idx = np.zeros(n_h, dtype='int')

    for i in range(n_h):
        for j in range(n_h):
            corr[i, j] = np.correlate(W_true[i, :], np.abs(W_trained[j, :]))

        shuffle_idx[i] = np.argmax(corr[i, :])


    if not allow_duplicates:

        # check for conflicts
        occurence_counts = np.zeros(n_h, dtype='int')

        for i in range(n_h):
            occurence_counts[i] += np.sum(shuffle_idx == i)

        # in the current data set, there are only every conflicts of 2 populations assigned to the same hidden unit
        assert (np.max(occurence_counts) <= 2)

        n_conflicts = np.sum(occurence_counts > 1)
        #if n_conflicts > 0:
        #    print("Found {} conflict(s)".format(n_conflicts))

        # resolve conflicts if there are any
        while n_conflicts > 0:
            # Any conflict in this data set is between two populations getting assigned the same hidden unit.
            # To solve this, the correlation with the hidden unit is checked for both populations;
            # The one with the highest correlation gets to keep its assignment, the other one
            # has to switch to a different hidden unit which has no assignments yet.
            # (If multiple options are still available, it chooses the one with the highest correlation)


            desired_assignment = np.where(occurence_counts == 2)[0][0] # look at first conflicting assignment
            leftover_assignments = np.where(occurence_counts == 0)[0] # left over options

            conflicting_populations = np.where(shuffle_idx == desired_assignment)[0] # will give two results

            loser = np.argmin(corr[conflicting_populations, desired_assignment]) # 0 or 1. Loser is the population with lower correlation  to desired assignment

            # loser switches its assignment to the best still available option
            new_loser_assignment = leftover_assignments[np.argmax(corr[conflicting_populations[loser], leftover_assignments])]
            shuffle_idx[conflicting_populations[loser]] = new_loser_assignment

            # update occerences
            occurence_counts[desired_assignment] -= 1
            occurence_counts[new_loser_assignment] += 1

            n_conflicts -= 1


    # Reshuffle matrices
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
    if not torch.is_tensor(data):
        data = torch.tensor(data)
    if n_batches == None:
        if data.dim() == 2:
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
    
    
def generate_moments(vt, vs_cRBM, vs_rtrbm, ht_cRBM, ht_rtrbm, hs_cRBM, hs_rtrbm, n_neurons=1000, idx=None):

    nv, T_batch, n_batches = vt.shape
    if idx is None:
        idx = np.random.permutation(nv)
    idx = idx[:n_neurons]
    
    vs_cRBM = vs_cRBM[idx, :]
    vs_cRBM = reshape(vs_cRBM, T=T_batch, n_batches=n_batches)
    vs_cRBM = reshape(vs_cRBM[:, 1:, :])
    vt = vt[idx, 1:, :] # 1: because in the sample function the RTRBM returns the sampled array without the first time step
    vs_rtrbm = vs_rtrbm[idx, :-1, :]  # :-1 because the sample function returns the sampled array without the first time step till T + 1
     
    ht_cRBM = reshape(ht_cRBM, T=T_batch, n_batches=n_batches)
    ht_cRBM = reshape(ht_cRBM[:, 1:, :])
    hs_cRBM = reshape(hs_cRBM, T=T_batch, n_batches=n_batches)
    hs_cRBM = reshape(hs_cRBM[:, 1:, :])
    ht_rtrbm = ht_rtrbm[:, 1:, :]
    hs_rtrbm = hs_rtrbm[:, :-1, :]

    nv, T_batch, n_batches = vt.shape
    T = T_batch * n_batches
    
    vvs_cRBM = torch.einsum('vT, VT -> vV', vs_cRBM, vs_cRBM) / T
    vvs_cRBM_t = torch.einsum('vT, VT -> vV', vs_cRBM[:, :-1], vs_cRBM[:, 1:]) / (T-1)

    hhs_cRBM_t = torch.einsum('hT, HT -> hH', hs_cRBM[:, :-1], hs_cRBM[:, 1:])/ (T-1)
    hht_cRBM_t = torch.einsum('hT, HT -> hH', ht_cRBM[:, :-1], ht_cRBM[:, 1:])/ (T-1)

    vvs_rtrbm, vvt, vvs_rtrbm_t, vvt_t, hhs_rtrbm_t, hht_rtrbm_t = 0, 0, 0, 0, 0, 0

    for batch in range(n_batches):
        vvs_rtrbm += torch.einsum('vT, VT -> vV', vs_rtrbm[..., batch], vs_rtrbm[..., batch]) / (T-1)
        vvt += torch.einsum('vT, VT -> vV', vt[..., batch], vt[..., batch]) / (T-1)

        vvs_rtrbm_t += torch.einsum('vT, VT -> vV', vs_rtrbm[:, :-1, batch], vs_rtrbm[:, 1:, batch]) / (T-1)
        vvt_t += torch.einsum('vT, VT -> vV', vt[:, :-1, batch], vt[:, 1:, batch]) / (T-1)

        hhs_rtrbm_t += torch.einsum('hT, HT -> hH', hs_rtrbm[:, :-1, batch], hs_rtrbm[:, 1:, batch]) / (T-1)
        hht_rtrbm_t += torch.einsum('hT, HT -> hH', ht_rtrbm[:, :-1, batch], ht_rtrbm[:, 1:, batch]) / (T-1)
    
    return vt.cpu(), vvt.cpu(), vvt_t.cpu(), hht_cRBM_t.cpu(), hht_rtrbm_t.cpu(), vs_cRBM.cpu(), vvs_cRBM.cpu(), vvs_cRBM_t.cpu(), hhs_cRBM_t.cpu(), vs_rtrbm.cpu(), vvs_rtrbm.cpu(), vvs_rtrbm_t.cpu(), hhs_rtrbm_t.cpu()

def compare_model_statistics(data_statistics, model_statistics_crbm, model_statistics_rtrbm, method='wilcoxon'):
    if len(data_statistics) == 2:
        difference_crbm = model_statistics_crbm - data_statistics[0]
        difference_rtrbm = model_statistics_rtrbm - data_statistics[1]
    else:
        difference_crbm = model_statistics_crbm - data_statistics
        difference_rtrbm = model_statistics_rtrbm - data_statistics
    
    if method == 'ranksums':
        r, pvalue = ranksums(difference_crbm**2, difference_rtrbm**2)
    elif method == 'kruskal':
        r, pvalue = kruskal(difference_crbm**2, difference_rtrbm**2)
    elif method == 't-test':
        r, pvalue = ttest_ind(difference_crbm**2, difference_rtrbm**2)
    elif method == 'wilcoxon':
        r, pvalue = wilcoxon(difference_crbm**2, difference_rtrbm**2)
    else:
        r, pvalue = ranksums(difference_crbm**2, difference_rtrbm**2)
        
    return r, pvalue
    
def compare_all_model_statistics(vt, vs_cRBM, vs_rtrbm, ht_cRBM, ht_rtrbm, hs_cRBM, hs_rtrbm, n_neurons):
    
    vt, vvt, vvt_t, hht_cRBM_t, hht_rtrbm_t, vs_cRBM, vvs_cRBM, vvs_cRBM_t, hhs_cRBM_t, vs_rtrbm, vvs_rtrbm, vvs_rtrbm_t, hhs_rtrbm_t = generate_moments(vt, vs_cRBM, vs_rtrbm, ht_cRBM, ht_rtrbm, hs_cRBM, hs_rtrbm, n_neurons=n_neurons)
    rs_v, p_v = compare_model_statistics(torch.mean(vt, (1,2)), torch.mean(vs_cRBM, 1), torch.mean(vs_rtrbm.cpu(), (1,2)))
    rs_vv, p_vv = compare_model_statistics(vvt.flatten(), vvs_cRBM.flatten(), vvs_rtrbm.cpu().flatten())
    rs_vvt, p_vvt = compare_model_statistics(vvt_t.flatten(), vvs_cRBM_t.flatten(), vvs_rtrbm_t.cpu().flatten())
    rs_hht, p_hht = compare_model_statistics([hht_cRBM_t.flatten(), hht_rtrbm_t.cpu().flatten()], hhs_cRBM_t.flatten(), hhs_rtrbm_t.cpu().flatten())
    return [rs_v, p_v, rs_vv, p_vv, rs_vvt, p_vvt, rs_hht, p_hht]

def density_scatter(x, y, ax=None, fig=None, r=None, vmax=None, n_bins=50, last=False, return_stats=True, add_text=True):
    if ax is None:
        fig, ax = plt.subplots()

    x, y = np.array(x), np.array(y)
    x[np.isnan(x)], y[np.isnan(y)]  = 0, 0
    hh = ax.hist2d(x, y, bins=(n_bins, n_bins), cmap=plt.get_cmap('bwr'), norm=LogNorm(vmin=1, vmax=vmax))

    if last:
        cbar_ax = fig.add_axes([0.93, 0.23, 0.01, 0.5])
        cbar = fig.colorbar(hh[3], pad=0.35, cax=cbar_ax, ticks=[1, vmax])
        cbar.minorticks_off()
        cbar.formatter = LogFormatterExponent(base=10)
        cbar.ax.set_ylabel('Log$_{10}$ PDF', fontsize=7)
        cbar.update_ticks()
        cbar.ax.set_yticklabels(['$0$', '$6$'], fontsize=7)
    else: cbar=0

    temp = y - x
    sSSD = np.sqrt(np.dot(temp.T, temp))
    if r is None:
        #r, pvalue = pearsonr(x, y)
        r, pvalue = spearmanr(a=x, b=y)
    if add_text is True:
        ax.text(.05, .9, '$r_s=%0.4s$'%(r), transform=ax.transAxes, fontsize=8, fontweight='heavy')
        ax.text(.05, .75, '$\\sqrt{SSD}=%0.4s$'%(sSSD), transform=ax.transAxes, fontsize=8, fontweight='heavy')

    if return_stats:
        return ax, cbar, r, pvalue
    else:
        return ax, cbar
                 
def plot_moments(vt, vs_cRBM, vs_rtrbm, ht_cRBM, ht_rtrbm, hs_cRBM, hs_rtrbm, n_neurons=10000, return_stats=False, add_text=True, n_bins=50):
    vt, vvt, vvt_t, hht_cRBM_t, hht_rtrbm_t, vs_cRBM, vvs_cRBM, vvs_cRBM_t, hhs_cRBM_t, vs_rtrbm, vvs_rtrbm, vvs_rtrbm_t, hhs_rtrbm_t = generate_moments(vt, vs_cRBM, vs_rtrbm, ht_cRBM, ht_rtrbm, hs_cRBM, hs_rtrbm, n_neurons=n_neurons)
    fs=8
    fig, axes = plt.subplots(2, 4, figsize=(1.4*6.3, 1.4*3))
    fig.subplots_adjust(right=0.9, wspace=0.5, hspace=0.5)

    ax = axes[0, 0]
    ax, cbar, rs_v_cRBM, ssd_v_cRBM = density_scatter(torch.mean(vt, (1, 2)), torch.mean(vs_cRBM, 1), ax=ax, fig=fig, vmax=1e6, n_bins=n_bins, add_text=add_text)
    ax.plot([0, 1], [0, 1], 'grey', linestyle='dotted', linewidth=1)
    ax.set_xlim([0, .6])
    ax.set_ylim([0, .6])
    ax.set_xticks([0, .6])
    ax.set_yticks([0, .6])
    ax.set_title('$\langle v_i \\rangle$', fontsize=fs)
    ax.tick_params(labelsize=7)

    ax = axes[0, 1]
    ax, cbar, rs_vv_cRBM, ssd_vv_cRBM = density_scatter(vvt.flatten(), vvs_cRBM.flatten(), ax=ax, fig=fig, vmax=1e6, n_bins=n_bins, add_text=add_text)
    ax.plot([0, 1], [0, 1], 'grey', linestyle='dotted', linewidth=1)
    ax.set_xlim([0, .6])
    ax.set_ylim([0, .6])
    ax.set_xticks([0, .6])
    ax.set_yticks([0, .6])
    ax.set_title('$\langle v_iv_j \\rangle$', fontsize=fs)
    ax.tick_params(labelsize=7)

    ax = axes[0, 2]
    ax, cbar, rs_vvt_cRBM, ssd_vvt_cRBM = density_scatter(vvt_t.flatten(), vvs_cRBM_t.flatten(), ax=ax, fig=fig, vmax=1e6, n_bins=n_bins, add_text=add_text)
    ax.plot([0, 1], [0, 1], 'grey', linestyle='dotted', linewidth=1)
    ax.set_xlim([0, .5])
    ax.set_ylim([0, .5])
    ax.set_xticks([0, .5])
    ax.set_yticks([0, .5])
    ax.set_title('$\langle v_i^{[t]}v_j^{[t+1]} \\rangle$', fontsize=fs)
    ax.tick_params(labelsize=7)

    ax = axes[0, 3]
    ax, cbar, rs_hht_cRBM, ssd_hht_cRBM = density_scatter(hht_cRBM_t.flatten(), hhs_cRBM_t.flatten(), ax=ax, fig=fig, vmax=1e6, n_bins=n_bins, add_text=add_text)
    ax.plot([-5, 5], [-5, 5], 'grey', linestyle='dotted', linewidth=1)
    ax.set_xlim([-3, torch.ceil(torch.max(hht_cRBM_t))])
    ax.set_ylim([-3, torch.ceil(torch.max(hhs_cRBM_t))])
    ax.set_xticks([-3, torch.floor(torch.max(hht_cRBM_t))])
    ax.set_yticks([-3, torch.floor(torch.max(hhs_cRBM_t))])
    ax.set_title('$\langle h_i^{[t]}h_j^{[t+1]} \\rangle$', fontsize=fs)
    ax.tick_params(labelsize=7)

    ax=axes[1, 0]
    ax, cbar, rs_v_rtrbm, ssd_v_rtrbm = density_scatter(torch.mean(vt, (1, 2)), torch.mean(vs_rtrbm.cpu(), (1, 2)), ax=ax, fig=fig, vmax=1e6, n_bins=n_bins, add_text=add_text)
    ax.plot([0, 1], [0, 1], 'grey', linestyle='dotted', linewidth=1)
    ax.set_xlim([0, .6])
    ax.set_ylim([0, .6])
    ax.set_xticks([0, .6])
    ax.set_yticks([0, .6])
    ax.set_title('$\langle v_i \\rangle$', fontsize=fs)
    ax.tick_params(labelsize=7)

    ax = axes[1, 1]
    ax, cbar, rs_vv_rtrbm, ssd_vv_rtrbm = density_scatter(vvt.flatten(), vvs_rtrbm.cpu().flatten(), ax=ax, fig=fig, vmax=1e6, n_bins=n_bins, add_text=add_text)
    ax.plot([0, 1], [0, 1], 'grey', linestyle='dotted', linewidth=1)
    # ax.plot(covt.flatten(), covs.flatten(), '.', markersize=1)
    ax.set_xlim([0, 0.6])
    ax.set_ylim([0, 0.6])
    ax.set_xticks([0, 0.6])
    ax.set_yticks([0, 0.6])
    ax.set_title('$\langle v_iv_j \\rangle$', fontsize=fs)
    ax.tick_params(labelsize=7)

    ax = axes[1, 2]
    ax, cbar, rs_vvt_rtrbm, ssd_vvt_rtrbm = density_scatter(vvt_t.flatten(), vvs_rtrbm_t.cpu().flatten(), ax=ax, fig=fig, vmax=1e6, n_bins=n_bins, add_text=add_text)
    ax.plot([0, 1], [0, 1], 'grey', linestyle='dotted', linewidth=1)
    ax.set_xlim([0, .5])
    ax.set_ylim([0, .5])
    ax.set_xticks([0, .5])
    ax.set_yticks([0, .5])
    ax.set_title('$\langle v_i^{[t]}v_j^{[t+1]} \\rangle$', fontsize=fs)
    ax.tick_params(labelsize=7)

    ax = axes[1, 3]
    ax, cbar, rs_hht_rtrbm, ssd_hht_rtrbm = density_scatter(hht_rtrbm_t.cpu().flatten(), hhs_rtrbm_t.cpu().flatten(), ax=ax, fig=fig, vmax=1e6, n_bins=n_bins, last=True, add_text=add_text)
    ax.plot([0, 2], [0, 2], 'grey', linestyle='dotted', linewidth=1)
    ax.set_xlim([0, 1.25])
    ax.set_ylim([0, 1.25])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_title('$\langle h_i^{[t]}h_j^{[t+1]} \\rangle$', fontsize=fs)
    ax.tick_params(labelsize=7)

    fig.suptitle('Moments', fontsize=fs, fontweight='bold')
    fig.supxlabel('Data statistics', fontsize=fs, fontweight='bold')
    fig.supylabel('Model statistics', fontsize=fs, fontweight='bold')
    #plt.show()

    if return_stats:
        return [rs_v_cRBM, rs_vv_cRBM, rs_vvt_cRBM, rs_hht_cRBM, rs_v_rtrbm, rs_vv_rtrbm, rs_vvt_rtrbm, rs_hht_rtrbm], [ssd_v_cRBM, ssd_vv_cRBM, ssd_vvt_cRBM, ssd_hht_cRBM, ssd_v_rtrbm, ssd_vv_rtrbm, ssd_vvt_rtrbm, ssd_hht_rtrbm]
    else: 
        return