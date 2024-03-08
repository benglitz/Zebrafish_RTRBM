import torch
import numpy as np
from itertools import permutations
from scipy.stats import pearsonr, ranksums, kruskal, wilcoxon, ttest_ind, spearmanr
from sklearn.mixture import GaussianMixture

from ..utils.data_methods import reshape_from_batches
from ..utils.data_methods import reshape


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


def generate_moments(vt, vs_cRBM, vs_rtrbm, ht_cRBM, ht_rtrbm, hs_cRBM, hs_rtrbm, n_neurons=1000, idx=None):
    nv, T_batch, n_batches = vt.shape
    if idx is None:
        idx = np.random.permutation(nv)
    idx = idx[:n_neurons]

    vs_cRBM = vs_cRBM[idx, :]
    vs_cRBM = reshape(vs_cRBM, T=T_batch, n_batches=n_batches)
    vs_cRBM = reshape(vs_cRBM[:, 1:, :])
    vt = vt[idx, 1:,
         :]  # 1: because in the sample function the RTRBM returns the sampled array without the first time step
    vs_rtrbm = vs_rtrbm[idx, :-1,
               :]  # :-1 because the sample function returns the sampled array without the first time step till T + 1

    ht_cRBM = reshape(ht_cRBM, T=T_batch, n_batches=n_batches)
    ht_cRBM = reshape(ht_cRBM[:, 1:, :])
    hs_cRBM = reshape(hs_cRBM, T=T_batch, n_batches=n_batches)
    hs_cRBM = reshape(hs_cRBM[:, 1:, :])
    ht_rtrbm = ht_rtrbm[:, 1:, :]
    hs_rtrbm = hs_rtrbm[:, :-1, :]

    nv, T_batch, n_batches = vt.shape
    T = T_batch * n_batches

    vvs_cRBM = torch.einsum('vT, VT -> vV', vs_cRBM, vs_cRBM) / T
    vvs_cRBM_t = torch.einsum('vT, VT -> vV', vs_cRBM[:, :-1], vs_cRBM[:, 1:]) / (T - 1)

    hhs_cRBM_t = torch.einsum('hT, HT -> hH', hs_cRBM[:, :-1], hs_cRBM[:, 1:]) / (T - 1)
    hht_cRBM_t = torch.einsum('hT, HT -> hH', ht_cRBM[:, :-1], ht_cRBM[:, 1:]) / (T - 1)

    vvs_rtrbm, vvt, vvs_rtrbm_t, vvt_t, hhs_rtrbm_t, hht_rtrbm_t = 0, 0, 0, 0, 0, 0

    for batch in range(n_batches):
        vvs_rtrbm += torch.einsum('vT, VT -> vV', vs_rtrbm[..., batch], vs_rtrbm[..., batch]) / (T - 1)
        vvt += torch.einsum('vT, VT -> vV', vt[..., batch], vt[..., batch]) / (T - 1)

        vvs_rtrbm_t += torch.einsum('vT, VT -> vV', vs_rtrbm[:, :-1, batch], vs_rtrbm[:, 1:, batch]) / (T - 1)
        vvt_t += torch.einsum('vT, VT -> vV', vt[:, :-1, batch], vt[:, 1:, batch]) / (T - 1)

        hhs_rtrbm_t += torch.einsum('hT, HT -> hH', hs_rtrbm[:, :-1, batch], hs_rtrbm[:, 1:, batch]) / (T - 1)
        hht_rtrbm_t += torch.einsum('hT, HT -> hH', ht_rtrbm[:, :-1, batch], ht_rtrbm[:, 1:, batch]) / (T - 1)

    return vt.cpu(), vvt.cpu(), vvt_t.cpu(), hht_cRBM_t.cpu(), hht_rtrbm_t.cpu(), vs_cRBM.cpu(), vvs_cRBM.cpu(), vvs_cRBM_t.cpu(), hhs_cRBM_t.cpu(), vs_rtrbm.cpu(), vvs_rtrbm.cpu(), vvs_rtrbm_t.cpu(), hhs_rtrbm_t.cpu()


def compare_model_statistics(data_statistics, model_statistics_crbm, model_statistics_rtrbm, method='wilcoxon'):
    if len(data_statistics) == 2:
        difference_crbm = model_statistics_crbm - data_statistics[0]
        difference_rtrbm = model_statistics_rtrbm - data_statistics[1]
    else:
        difference_crbm = model_statistics_crbm - data_statistics
        difference_rtrbm = model_statistics_rtrbm - data_statistics

    if method == 'ranksums':
        r, pvalue = ranksums(difference_crbm ** 2, difference_rtrbm ** 2)
    elif method == 'kruskal':
        r, pvalue = kruskal(difference_crbm ** 2, difference_rtrbm ** 2)
    elif method == 't-test':
        r, pvalue = ttest_ind(difference_crbm ** 2, difference_rtrbm ** 2)
    elif method == 'wilcoxon':
        r, pvalue = wilcoxon(difference_crbm ** 2, difference_rtrbm ** 2)
    else:
        r, pvalue = ranksums(difference_crbm ** 2, difference_rtrbm ** 2)

    return r, pvalue


def compare_all_model_statistics(vt, vs_cRBM, vs_rtrbm, ht_cRBM, ht_rtrbm, hs_cRBM, hs_rtrbm, n_neurons):
    vt, vvt, vvt_t, hht_cRBM_t, hht_rtrbm_t, vs_cRBM, vvs_cRBM, vvs_cRBM_t, hhs_cRBM_t, vs_rtrbm, vvs_rtrbm, vvs_rtrbm_t, hhs_rtrbm_t = generate_moments(
        vt, vs_cRBM, vs_rtrbm, ht_cRBM, ht_rtrbm, hs_cRBM, hs_rtrbm, n_neurons=n_neurons)
    rs_v, p_v = compare_model_statistics(torch.mean(vt, (1, 2)), torch.mean(vs_cRBM, 1),
                                         torch.mean(vs_rtrbm.cpu(), (1, 2)))
    rs_vv, p_vv = compare_model_statistics(vvt.flatten(), vvs_cRBM.flatten(), vvs_rtrbm.cpu().flatten())
    rs_vvt, p_vvt = compare_model_statistics(vvt_t.flatten(), vvs_cRBM_t.flatten(), vvs_rtrbm_t.cpu().flatten())
    rs_hht, p_hht = compare_model_statistics([hht_cRBM_t.flatten(), hht_rtrbm_t.cpu().flatten()], hhs_cRBM_t.flatten(),
                                             hhs_rtrbm_t.cpu().flatten())
    return [rs_v, p_v, rs_vv, p_vv, rs_vvt, p_vvt, rs_hht, p_hht]

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


def get_spearmanr(vt, vs_cRBM, vs_rtrbm, ht_cRBM, ht_rtrbm, hs_cRBM, hs_rtrbm, n_neurons=1000, idx=None):
    vt, vvt, vvt_t, hht_cRBM_t, hht_rtrbm_t, vs_cRBM, vvs_cRBM, vvs_cRBM_t, hhs_cRBM_t, vs_rtrbm, vvs_rtrbm, vvs_rtrbm_t, hhs_rtrbm_t = generate_moments(
        vt, vs_cRBM, vs_rtrbm, ht_cRBM, ht_rtrbm, hs_cRBM, hs_rtrbm, n_neurons=n_neurons, idx=idx)

    rs_v_cRBM, ssd_v_cRBM = spearmanr(torch.mean(vt, (1, 2)), torch.mean(vs_cRBM, 1))
    rs_vv_cRBM, ssd_vv_cRBM = spearmanr(vvt.flatten(), vvs_cRBM.flatten())
    rs_vvt_cRBM, ssd_vvt_cRBM = spearmanr(vvt_t.flatten(), vvs_cRBM_t.flatten())
    rs_hht_cRBM, ssd_hht_cRBM = spearmanr(hht_cRBM_t.flatten(), hhs_cRBM_t.flatten())

    rs_v_rtrbm, ssd_v_rtrbm = spearmanr(torch.mean(vt, (1, 2)), torch.mean(vs_rtrbm.cpu(), (1, 2)))
    rs_vv_rtrbm, ssd_vv_rtrbm = spearmanr(vvt.flatten(), vvs_rtrbm.cpu().flatten())
    rs_vvt_rtrbm, ssd_vvt_rtrbm = spearmanr(vvt_t.flatten(), vvs_rtrbm_t.cpu().flatten())
    rs_hht_rtrbm, ssd_hht_rtrbm = spearmanr(hht_rtrbm_t.cpu().flatten(), hhs_rtrbm_t.cpu().flatten())

    return [rs_v_cRBM, rs_vv_cRBM, rs_vvt_cRBM, rs_hht_cRBM, rs_v_rtrbm, rs_vv_rtrbm, rs_vvt_rtrbm, rs_hht_rtrbm], [
        ssd_v_cRBM, ssd_vv_cRBM, ssd_vvt_cRBM, ssd_hht_cRBM, ssd_v_rtrbm, ssd_vv_rtrbm, ssd_vvt_rtrbm, ssd_hht_rtrbm]


def fit_gmm_per_t(H):
    n_h, T = H.shape
    means = np.zeros((T, 2))
    for t in range(T):
        gmm = GaussianMixture(n_components=2)
        gmm.fit(H[:, t].reshape(-1, 1))
        means_t = gmm.means_.flatten()
        means[t, :] = np.sort(means_t)
    return means


def kurtosis_per_t(H):
    if H.ndim == 3:
        H = reshape(H)
    n_h, T = H.shape
    mean_left_peak = fit_gmm_per_t(H)[:, 0]
    H = H - mean_left_peak
    try:
        PR = np.sum(H ** 2, 0) ** 2 / np.sum(H ** 4, 0)
    except:
        PR = torch.sum(H ** 2, 0) ** 2 / torch.sum(H ** 4, 0)
    return PR / n_h
