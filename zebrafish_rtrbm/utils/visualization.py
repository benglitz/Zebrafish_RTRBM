import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterExponent
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr

from zebrafish_rtrbm.utils.metrics import generate_moments


def optimal_cluster_plot(X, n_clusters=10):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
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


def density_scatter(x, y, ax=None, fig=None, r=None, vmax=None, n_bins=50, last=False, return_stats=True,
                    add_text=True):
    if ax is None:
        fig, ax = plt.subplots()

    x, y = np.array(x), np.array(y)
    x[np.isnan(x)], y[np.isnan(y)] = 0, 0
    hh = ax.hist2d(x, y, bins=(n_bins, n_bins), cmap=plt.get_cmap('bwr'), norm=LogNorm(vmin=1, vmax=vmax))

    if last:
        cbar_ax = fig.add_axes([0.93, 0.23, 0.01, 0.5])
        cbar = fig.colorbar(hh[3], pad=0.35, cax=cbar_ax, ticks=[1, vmax])
        cbar.minorticks_off()
        cbar.formatter = LogFormatterExponent(base=10)
        cbar.ax.set_ylabel('Log$_{10}$ PDF', fontsize=7)
        cbar.update_ticks()
        cbar.ax.set_yticklabels(['$0$', '$6$'], fontsize=7)
    else:
        cbar = 0

    temp = y - x
    sSSD = np.sqrt(np.dot(temp.T, temp))
    if r is None:
        # r, pvalue = pearsonr(x, y)
        r, pvalue = spearmanr(a=x, b=y)
    if add_text is True:
        ax.text(.05, .9, '$r_s=%0.4s$' % (r), transform=ax.transAxes, fontsize=8, fontweight='heavy')
        ax.text(.05, .75, '$\\sqrt{SSD}=%0.4s$' % (sSSD), transform=ax.transAxes, fontsize=8, fontweight='heavy')

    if return_stats:
        return ax, cbar, r, pvalue
    else:
        return ax, cbar


def plot_moments(vt, vs_cRBM, vs_rtrbm, ht_cRBM, ht_rtrbm, hs_cRBM, hs_rtrbm, n_neurons=10000, return_stats=False,
                 add_text=True, n_bins=50):
    vt, vvt, vvt_t, hht_cRBM_t, hht_rtrbm_t, vs_cRBM, vvs_cRBM, vvs_cRBM_t, hhs_cRBM_t, vs_rtrbm, vvs_rtrbm,\
    vvs_rtrbm_t, hhs_rtrbm_t = generate_moments(
        vt, vs_cRBM, vs_rtrbm, ht_cRBM, ht_rtrbm, hs_cRBM, hs_rtrbm, n_neurons=n_neurons)
    fs = 8
    fig, axes = plt.subplots(2, 4, figsize=(1.4 * 6.3, 1.4 * 3))
    fig.subplots_adjust(right=0.9, wspace=0.5, hspace=0.5)

    ax = axes[0, 0]
    ax, cbar, rs_v_cRBM, ssd_v_cRBM = density_scatter(torch.mean(vt, (1, 2)), torch.mean(vs_cRBM, 1), ax=ax, fig=fig,
                                                      vmax=1e6, n_bins=n_bins, add_text=add_text)
    ax.plot([0, 1], [0, 1], 'grey', linestyle='dotted', linewidth=1)
    ax.set_xlim([0, .6])
    ax.set_ylim([0, .6])
    ax.set_xticks([0, .6])
    ax.set_yticks([0, .6])
    ax.set_title('$\langle v_i \\rangle$', fontsize=fs)
    ax.tick_params(labelsize=7)

    ax = axes[0, 1]
    ax, cbar, rs_vv_cRBM, ssd_vv_cRBM = density_scatter(vvt.flatten(), vvs_cRBM.flatten(), ax=ax, fig=fig, vmax=1e6,
                                                        n_bins=n_bins, add_text=add_text)
    ax.plot([0, 1], [0, 1], 'grey', linestyle='dotted', linewidth=1)
    ax.set_xlim([0, .6])
    ax.set_ylim([0, .6])
    ax.set_xticks([0, .6])
    ax.set_yticks([0, .6])
    ax.set_title('$\langle v_iv_j \\rangle$', fontsize=fs)
    ax.tick_params(labelsize=7)

    ax = axes[0, 2]
    ax, cbar, rs_vvt_cRBM, ssd_vvt_cRBM = density_scatter(vvt_t.flatten(), vvs_cRBM_t.flatten(), ax=ax, fig=fig,
                                                          vmax=1e6, n_bins=n_bins, add_text=add_text)
    ax.plot([0, 1], [0, 1], 'grey', linestyle='dotted', linewidth=1)
    ax.set_xlim([0, .5])
    ax.set_ylim([0, .5])
    ax.set_xticks([0, .5])
    ax.set_yticks([0, .5])
    ax.set_title('$\langle v_i^{[t]}v_j^{[t+1]} \\rangle$', fontsize=fs)
    ax.tick_params(labelsize=7)

    ax = axes[0, 3]
    ax, cbar, rs_hht_cRBM, ssd_hht_cRBM = density_scatter(hht_cRBM_t.flatten(), hhs_cRBM_t.flatten(), ax=ax, fig=fig,
                                                          vmax=1e6, n_bins=n_bins, add_text=add_text)
    ax.plot([-5, 5], [-5, 5], 'grey', linestyle='dotted', linewidth=1)
    ax.set_xlim([-3, torch.ceil(torch.max(hht_cRBM_t))])
    ax.set_ylim([-3, torch.ceil(torch.max(hhs_cRBM_t))])
    ax.set_xticks([-3, torch.floor(torch.max(hht_cRBM_t))])
    ax.set_yticks([-3, torch.floor(torch.max(hhs_cRBM_t))])
    ax.set_title('$\langle h_i^{[t]}h_j^{[t+1]} \\rangle$', fontsize=fs)
    ax.tick_params(labelsize=7)

    ax = axes[1, 0]
    ax, cbar, rs_v_rtrbm, ssd_v_rtrbm = density_scatter(torch.mean(vt, (1, 2)), torch.mean(vs_rtrbm.cpu(), (1, 2)),
                                                        ax=ax, fig=fig, vmax=1e6, n_bins=n_bins, add_text=add_text)
    ax.plot([0, 1], [0, 1], 'grey', linestyle='dotted', linewidth=1)
    ax.set_xlim([0, .6])
    ax.set_ylim([0, .6])
    ax.set_xticks([0, .6])
    ax.set_yticks([0, .6])
    ax.set_title('$\langle v_i \\rangle$', fontsize=fs)
    ax.tick_params(labelsize=7)

    ax = axes[1, 1]
    ax, cbar, rs_vv_rtrbm, ssd_vv_rtrbm = density_scatter(vvt.flatten(), vvs_rtrbm.cpu().flatten(), ax=ax, fig=fig,
                                                          vmax=1e6, n_bins=n_bins, add_text=add_text)
    ax.plot([0, 1], [0, 1], 'grey', linestyle='dotted', linewidth=1)
    # ax.plot(covt.flatten(), covs.flatten(), '.', markersize=1)
    ax.set_xlim([0, 0.6])
    ax.set_ylim([0, 0.6])
    ax.set_xticks([0, 0.6])
    ax.set_yticks([0, 0.6])
    ax.set_title('$\langle v_iv_j \\rangle$', fontsize=fs)
    ax.tick_params(labelsize=7)

    ax = axes[1, 2]
    ax, cbar, rs_vvt_rtrbm, ssd_vvt_rtrbm = density_scatter(vvt_t.flatten(), vvs_rtrbm_t.cpu().flatten(), ax=ax,
                                                            fig=fig, vmax=1e6, n_bins=n_bins, add_text=add_text)
    ax.plot([0, 1], [0, 1], 'grey', linestyle='dotted', linewidth=1)
    ax.set_xlim([0, .5])
    ax.set_ylim([0, .5])
    ax.set_xticks([0, .5])
    ax.set_yticks([0, .5])
    ax.set_title('$\langle v_i^{[t]}v_j^{[t+1]} \\rangle$', fontsize=fs)
    ax.tick_params(labelsize=7)

    ax = axes[1, 3]
    ax, cbar, rs_hht_rtrbm, ssd_hht_rtrbm = density_scatter(hht_rtrbm_t.cpu().flatten(), hhs_rtrbm_t.cpu().flatten(),
                                                            ax=ax, fig=fig, vmax=1e6, n_bins=n_bins, last=True,
                                                            add_text=add_text)
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
    # plt.show()

    if return_stats:
        return [rs_v_cRBM, rs_vv_cRBM, rs_vvt_cRBM, rs_hht_cRBM, rs_v_rtrbm, rs_vv_rtrbm, rs_vvt_rtrbm, rs_hht_rtrbm], [
            ssd_v_cRBM, ssd_vv_cRBM, ssd_vvt_cRBM, ssd_hht_cRBM, ssd_v_rtrbm, ssd_vv_rtrbm, ssd_vvt_rtrbm,
            ssd_hht_rtrbm]
    else:
        return