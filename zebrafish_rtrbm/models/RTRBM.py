"""
Basic implementation of the recurrent temporal restricted boltzmann machine[1].

~ Sebastian Quiroz Monnens & Casper Peters
References
----------
[1] : Sutskever, I., Hinton, G.E., & Taylor, G. (2008) The Recurrent Temporal Restricted Boltzmann Machine
"""

import torch
import numpy as np
from tqdm import tqdm
from utils.lr_scheduler import get_lrs


class RTRBM(object):
    def __init__(self, data: torch.Tensor, n_hidden: int = 10, device: str = 'cpu', debug_mode: bool = False):

        '''
        This function initializes the parameters of the Recurrent Temporal-RBM.

        Parameters
        ----------
        data : torch.Tensor
            The data to be modeled.
            Shape: (n_visible, T, n_batch) or (n_visible, T)
        n_hidden : int
            The number of hidden units.
        device : str
            The device to be used for computations.
        debug_mode : bool
            If True, the parameter history is stored.

        Returns
        -------
        None
        '''

        if not torch.cuda.is_available():
            print('cuda not available, using cpu')
            self.device = 'cpu'
        else:
            self.device = device
        self.V = data
        if self.V.ndim == 2:
            self.V = self.V[..., None]
        self.n_visible, self.T, self.num_samples = self.V.shape
        if self.V.ndim != 3:
            raise ValueError(
                "Data is not correctly defined: Use (n_visible, T) or (n_visible, T, num_samples) dimensions")

        self.n_hidden = n_hidden
        self.W = 0.01 * torch.randn(self.n_hidden, self.n_visible, dtype=torch.float, device=self.device)
        self.U = 0.01 * torch.randn(self.n_hidden, self.n_hidden, dtype=torch.float, device=self.device)
        self.b_h = torch.zeros(self.n_hidden, dtype=torch.float, device=self.device)
        self.b_v = torch.zeros(self.n_visible, dtype=torch.float, device=self.device)
        self.b_init = torch.zeros(self.n_hidden, dtype=torch.float, device=self.device)
        self.params = [self.W, self.U, self.b_h, self.b_v, self.b_init]
        self.debug_mode = debug_mode
        if debug_mode:
            self.parameter_history = []
        self.Dparams = self.initialize_grad_updates()
        self.errors = []

    def _parallel_recurrent_sample_r_given_v(self, v: torch.Tensor) -> torch.Tensor:
        '''

        Sample from the conditional distribution the expected hidden units given the visible units in parallel over batches.

        Parameters
        ----------
        v : torch.Tensor
            The visible units.
            Shape: (n_visible, T, n_batch)

        Returns
        -------
        r : torch.Tensor
            The expected hiddens units.
            Shape: (n_hiddens, T, n_batch)
        '''

        _, T, n_batch = v.shape
        r = torch.zeros(self.n_hidden, T, n_batch, device=v.device)
        r[:, 0, :] = torch.sigmoid(torch.einsum('hv, vb -> hb', self.W, v[:, 0, :]) + self.b_init[:, None])
        for t in range(1, T):
            r[:, t, :] = torch.sigmoid(torch.einsum('hv, vb -> hb', self.W, v[:, t, :]) + \
                                       torch.einsum('hr, rb -> hb', self.U, r[:, t - 1, :]) + self.b_h[:, None])
        return r

    def _parallel_sample_r_h_given_v(self, v: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        '''
        Sample from the conditional distribution the expected hidden- and hidden units given the visible units in parallel over time and batches.

        Parameters
        ----------
        v : torch.Tensor
            The visible units.
            Shape: (n_visible, T, n_batch)

        Returns
        -------
        r : torch.Tensor
            The expected hiddens units.
            Shape: (n_hiddens, T, n_batch)

        h : torch.Tensor
            The hiddens units.
            Shape: (n_hiddens, T, n_batch)
        '''
        r0 = torch.sigmoid(torch.einsum('hv, vb->hb', self.W, v[:, 0, :]) + self.b_init[:, None])
        r = torch.sigmoid(torch.einsum('hv, vTb->hTb', self.W, v[:, 1:, :]) + \
                          torch.einsum('hr, rTb->hTb', self.U, r[:, :-1, :]) + self.b_h[:, None, None])
        r = torch.cat([r0[:, None, :], r], 1)
        return r, torch.bernoulli(r)

    def _parallel_sample_v_given_h(self, h: torch.Tensor) -> torch.Tensor:
        '''
        Sample from the coditional distribution the visible units given the hidden units in parallel over batches.

        Parameters
        ----------
        h : torch.Tensor
            The hidden units.
            Shape: (n_hidden, T, n_batch)

        Returns
        -------
        v : torch.Tensor
            The visible units.
            Shape: (n_visible, T, n_batch)
        '''

        v_prob = torch.sigmoid(torch.einsum('hv, hTb->vTb', self.W, h) + self.b_v[:, None, None])
        return torch.bernoulli(v_prob)

    def CD(self, v_data: torch.Tensor, r_data: torch.Tensor, CDk: int = 1) -> torch.Tensor:
        '''
        Perform Contrastive Divergence (CD) for a all time steps and batches in parallel.

        Parameters
        ----------
        v_data : torch.Tensor
            The visible units.
            Shape: (n_visible, T, n_batch)
        r_data: torch.Tensor
            The expected hidden units.
            Shape: (n_hidden, T, n_batch)
        CDk : int
            The number of steps of CD to perform.

        Returns
        -------
        torch.mean(probht_k, 3) : torch.Tensor
            The mean of the expected hidden units over all CD steps.
            Shape: (n_hidden, T, n_batch)
        torch.mean(vt_k, 3) : torch.Tensor
            The mean of the visible units over all CD steps.
            Shape: (n_visible, T, n_batch)
        ht_k : torch.Tensor
            Sampled hidden units of all CD steps.
            Shape: (n_hidden, T, n_batch, CDk)
        vt_k : torch.Tensor
            Sampled visible units of all CD steps.
            Shape: (n_visible, T, n_batch, CDk)
        '''

        batchsize = v_data.shape[2]
        ht_k = torch.zeros(self.n_hidden, self.T, batchsize, CDk, dtype=torch.float, device=self.device)
        probht_k = torch.zeros(self.n_hidden, self.T, batchsize, CDk, dtype=torch.float, device=self.device)
        vt_k = torch.zeros(self.n_visible, self.T, batchsize, CDk, dtype=torch.float, device=self.device)
        vt_k[..., 0] = v_data.detach().clone()
        probht_k[..., 0], ht_k[..., 0] = self._parallel_sample_r_h_given_v(v_data, r_data)
        for i in range(1, CDk):
            vt_k[..., i] = self._parallel_sample_v_given_h(ht_k[..., i - 1])
            probht_k[..., i], ht_k[..., i] = self._parallel_sample_r_h_given_v(vt_k[..., i], r_data)
        return torch.mean(probht_k, 3), torch.mean(vt_k, 3), ht_k, vt_k

    def learn(self, n_epochs=1000,
              lr=None,
              lr_schedule=None,
              batch_size=1,
              CDk=10,
              PCD=False,
              sp=None, x=2,
              mom=0, wc=0,
              disable_tqdm=False,
              save_every_n_epochs=1, shuffle_batch=True, n=1,
              **kwargs):
        '''
        Train the RTRBM.

        Parameters
        ----------
        n_epochs : int
            Number of epochs to train for.
        lr : float
            Learning rate.
        lr_schedule : str
            Learning rate schedule; 'cyclic', 'geometric_decay', 'linear_decay', 'cosine_annealing_warm_restarts', 'cyclic_annealing'
        batch_size : int
            Number of batches to evaluate before updating gradients
        CDk : int
            The number of steps of CD to perform.
        PCD : bool
            Perform persistent contrastive divergence
        sp : float
            Sparse penalty; regularization penalty for the hidden to visible weights for not being sparse
            ~ sp * abs(W_ij)^(x-1) * sign(W(ij))
        x : int
            Sparsity exponent to determine magnitude of the sparse penalty
        mom : float
            Momentum
        wc : float
            Weight cost
        disable_tqdm: bool
            disable load bar for training RTRBM
        save_every_n_epochs: int
            Save every n epochs the parameters of the RTRBM
        shuffle_batch: bool
            Shuffle the batches each epoch so that stochasticity is optimized during learning
        kwargs: Model
            Learning parameters corresponding to learning rate schedule (lr_schedule)

        Returns
        -------
        None

        '''

        if self.num_samples <= batch_size:
            batch_size, num_batches = self.num_samples, 1
        else:
            num_batches = self.num_samples // batch_size

        if lr is None:
            lrs = np.array(get_lrs(mode=lr_schedule, n_epochs=n_epochs, **kwargs))
        else:
            lrs = lr * torch.ones(n_epochs)

        self.disable = disable_tqdm
        self.lrs = lrs
        for epoch in tqdm(range(0, n_epochs), disable=self.disable):
            err = 0
            for batch in range(0, num_batches):
                self.dparams = self.initialize_grad_updates()
                v_data = self.V[:, :, batch * batch_size: (batch + 1) * batch_size].to(self.device)
                r_data = self._parallel_recurrent_sample_r_given_v(v_data)

                if PCD and epoch != 0:
                    barht, barvt, ht_k, vt_k = self.CD(vt_k[:, :, -1], CDk)
                else:
                    barht, barvt, ht_k, vt_k = self.CD(v_data, r_data, CDk)

                err += torch.sum((v_data - vt_k[..., -1]) ** 2).cpu()
                self.dparams = self.grad(v_data, r_data, ht_k, vt_k, barvt, barht)

                self.update_grad(lr=lrs[epoch], mom=mom, wc=wc, sp=sp, x=x, n=n)

            self.errors += [err / self.V.numel()]
            if self.debug_mode and epoch % save_every_n_epochs == 0:
                self.parameter_history.append([param.detach().clone().cpu() for param in self.params])
            if shuffle_batch:
                self.V[..., :] = self.V[..., torch.randperm(self.num_samples)]

    def initialize_grad_updates(self):
        '''
        Initializes the gradient updates for each parameter in the model.

        Parameters
        ----------
        self : Model
            The model to initialize the gradient updates for.

        Returns
        -------
        list
        A list of zeros of the same shape as each parameter in the model.
        '''
        return [torch.zeros_like(param, dtype=torch.float, device=self.device) for param in self.params]

    def grad(self, v_data: torch.Tensor, r_data: torch.Tensor, ht_k: torch.Tensor, vt_k: torch.Tensor,
             barvt: torch.Tensor, barht: torch.Tensor) -> torch.Tensor:
        '''

        Computes the gradients of the log-likelihood with respect to the parameters of the model.

        Parameters
        ----------
        v_data : torch.Tensor
            The visible units.
            Shape: (n_visible, T, n_batch)
        r_data: torch.Tensor
            The expected hidden units.
            Shape: (n_hidden, T, n_batch)
        ht_k : torch.Tensor
            Sampled hidden units of all CD steps.
            Shape: (n_hidden, T, n_batch, CDk)
        vt_k : torch.Tensor
            Sampled visible units of all CD steps.
            Shape: (n_visible, T, n_batch, CDk)
        barht : torch.Tensor
            The mean of the expected hidden units over all CD steps.
            Shape: (n_hidden, T, n_batch)
        barvt : torch.Tensor
            The mean of the visible units over all CD steps.
            Shape: (n_visible, T, n_batch)

        Returns
        -------
        list
        A list of gradients of the model parameters
        '''
        Dt = torch.zeros(self.n_hidden, self.T + 1, v_data.shape[2], dtype=torch.float, device=self.device)
        for t in range(self.T - 1, -1, -1):
            Dt[:, t, :] = torch.einsum('hv, hb->vb', self.U,
                                       (Dt[:, t + 1, :] * r_data[:, t, :] * (1 - r_data[:, t, :]) + \
                                        (r_data[:, t, :] - barht[:, t, :])))

        db_init = torch.mean((r_data[:, 0, :] - barht[:, 0, :]) + Dt[:, 1, :] * r_data[:, 0, :] * (1 - r_data[:, 0, :]),
                             1)
        tmp = torch.sum(Dt[:, 2:self.T, :] * (r_data[:, 1:self.T - 1, :] * (1 - r_data[:, 1:self.T - 1, :])), 1)
        db_H = torch.mean(torch.sum(r_data[:, 1:self.T, :], 1) - torch.sum(barht[:, 1:self.T, :], 1) + tmp, 1)
        db_V = torch.mean(torch.sum(v_data - barvt, 1), 1)
        dW = torch.mean(torch.einsum('rTb, vTb -> rvb',
                                     Dt[:, 1:self.T, :] * r_data[:, 0:self.T - 1, :] * (1 - r_data[:, 0:self.T - 1, :]),
                                     v_data[:, 0:self.T - 1, :]), 2)
        dW += torch.mean(
            torch.einsum('rTb, vTb -> rvb', r_data, v_data) - torch.mean(torch.einsum('rTbk, vTbk -> rvbk', ht_k, vt_k),
                                                                         3), 2)
        dU = torch.mean(torch.einsum('rTb, hTb -> rhb', Dt[:, 2:self.T + 1, :] * (
                    r_data[:, 1:self.T, :] * (1 - r_data[:, 1:self.T, :])) + r_data[:, 1:self.T, :] - barht[:, 1:self.T,
                                                                                                      :],
                                     r_data[:, 0:self.T - 1, :]), 2)
        return [dW, dU, db_H, db_V, db_init]

    def update_grad(self, lr=1e-3, mom=0, wc=0, sp=None, x=2, n=1):
        '''
        Updates the parameters of the RTRBM using the gradients computed in grad.

        Parameters
        ----------
        lr : float
            Learning rate
        mom : float
            Momentum
        wc : float
            Weight cost
        sp : float
            Sparse penalty; regularization penalty for the hidden to visible weights for not being sparse
        x : float
            Sparsity exponent to determine magnitude of the sparse penalty

        Returns
        -------
        None
        '''
        dW, dU, db_h, db_v, db_init = self.dparams
        DW, DU, Db_h, Db_v, Db_init = self.Dparams
        DW = mom * DW + lr / n * (dW - wc * self.W)
        DU = mom * DU + lr * (dU - wc * self.U)
        Db_h = mom * Db_h + lr * db_h
        Db_v = mom * Db_v + lr * db_v
        Db_init = mom * Db_init + lr * db_init
        if sp is not None:
            DW -= sp * torch.reshape(torch.sum(torch.abs(self.W), 1).repeat(self.n_visible),
                                     [self.n_hidden, self.n_visible]) ** (x - 1) * torch.sign(self.W)
        self.Dparams = [DW, DU, Db_h, Db_v, Db_init]
        for i in range(len(self.params)): self.params[i] += self.Dparams[i]
        return

    def return_params(self):
        '''
        This function returns the parameters of the RTRBM.

        Parameters
        ----------
        self : Model
            The RTRBM object.

        Returns
        -------
        list
        A list of the parameters of the RTRBM.
        '''
        return [self.W, self.U, self.b_v, self.b_init, self.b_h, self.errors]

    def infer(self,
              data,
              AF=torch.sigmoid,
              pre_gibbs_k=50,
              gibbs_k=10,
              mode=2,
              t_extra=0,
              disable_tqdm=False):

        T = self.T
        n_hidden = self.n_hidden
        n_visible, t1 = data.shape

        vt = torch.zeros(n_visible, T + t_extra, dtype=torch.float, device=self.device)
        rt = torch.zeros(n_hidden, T + t_extra, dtype=torch.float, device=self.device)
        vt[:, 0:t1] = data.float().to(self.device)

        rt[:, 0] = AF(torch.matmul(self.W, vt[:, 0]) + self.b_init)
        for t in range(1, t1):
            rt[:, t] = AF(torch.matmul(self.W, vt[:, t]) + self.b_h + torch.matmul(self.U, rt[:, t - 1]))

        for t in tqdm(range(t1, T + t_extra), disable=disable_tqdm):
            v = vt[:, t - 1]

            for kk in range(pre_gibbs_k):
                h = torch.bernoulli(AF(torch.matmul(self.W, v).T + self.b_h + torch.matmul(self.U, rt[:, t - 1]))).T
                v = torch.bernoulli(AF(torch.matmul(self.W.T, h) + self.b_v.T))

            vt_k = torch.zeros(n_visible, gibbs_k, dtype=torch.float, device=self.device)
            ht_k = torch.zeros(n_hidden, gibbs_k, dtype=torch.float, device=self.device)
            for kk in range(gibbs_k):
                h = torch.bernoulli(AF(torch.matmul(self.W, v).T + self.b_h + torch.matmul(self.U, rt[:, t - 1]))).T
                v = torch.bernoulli(AF(torch.matmul(self.W.T, h) + self.b_v.T))
                vt_k[:, kk] = v.T
                ht_k[:, kk] = h.T

            if mode == 1:
                vt[:, t] = vt_k[:, -1]
            if mode == 2:
                vt[:, t] = torch.mean(vt_k, 1)
            if mode == 3:
                E = torch.sum(ht_k * (torch.matmul(self.W, vt_k)), 0) + torch.matmul(self.b_v, vt_k) + torch.matmul(
                    self.b_h, ht_k) + torch.matmul(torch.matmul(self.U, rt[:, t - 1]).T, ht_k)
                idx = torch.argmax(E)
                vt[:, t] = vt_k[:, idx]

            rt[:, t] = AF(torch.matmul(self.W, vt[:, t]) + self.b_h + torch.matmul(self.U, rt[:, t - 1]))

        return vt, rt

    def sample(self, v_start, chain=50, pre_gibbs_k=100, gibbs_k=20, mode=1, disable_tqdm=False):
        n_hidden, n_visible = self.W.shape
        if v_start.ndim == 1: v_start = v_start[:, None]
        n_batches = v_start.shape[1]
        vt = torch.zeros(n_visible, chain + 1, n_batches, dtype=torch.float, device=self.device)
        rt = torch.zeros(n_hidden, chain + 1, n_batches, dtype=torch.float, device=self.device)

        rt[:, 0, :] = torch.sigmoid(torch.einsum('hv, vb->hb', self.W, v_start) + self.b_init[:, None])
        vt[:, 0, :] = v_start
        for t in tqdm(range(1, chain + 1), disable=disable_tqdm):
            v = vt[:, t - 1, :]
            for kk in range(pre_gibbs_k):
                h = torch.bernoulli(torch.sigmoid(torch.einsum('hv, vb->hb', self.W, v) + \
                                                  torch.einsum('hr, rb->hb', self.U, rt[:, t - 1, :]) + self.b_h[:,
                                                                                                        None]))
                v = torch.bernoulli(torch.sigmoid(torch.einsum('hv, hb->vb', self.W, h) + self.b_v[:, None]))

            vt_k = torch.zeros(n_visible, gibbs_k, n_batches, dtype=torch.float, device=self.device)
            ht_k = torch.zeros(n_hidden, gibbs_k, n_batches, dtype=torch.float, device=self.device)
            for kk in range(gibbs_k):
                h = torch.bernoulli(torch.sigmoid(torch.einsum('hv, vb->hb', self.W, v) + \
                                                  torch.einsum('hr, rb->hb', self.U, rt[:, t - 1, :]) + self.b_h[:,
                                                                                                        None]))
                v = torch.bernoulli(torch.sigmoid(torch.einsum('hv, hb->vb', self.W, h) + self.b_v[:, None]))
                vt_k[:, kk, :] = v
                ht_k[:, kk, :] = h

            if mode == 1:
                vt[:, t] = vt_k[:, -1]
            if mode == 2:
                vt[:, t] = torch.mean(vt_k, 1)
            if mode == 3:
                E = torch.sum(ht_k * (torch.matmul(self.W, vt_k)), 0) + torch.matmul(self.b_v, vt_k) + torch.matmul(
                    self.b_h, ht_k) + torch.matmul(torch.matmul(self.U, rt[:, t - 1]).T, ht_k)
                idx = torch.argmax(E)
                vt[:, t] = vt_k[:, idx]

            rt[:, t, :] = torch.sigmoid(torch.einsum('hv, vb->hb', self.W, v) + \
                                        torch.einsum('hr, rb->hb', self.U, rt[:, t - 1, :]) + self.b_h[:, None])

        return vt[:, 1:, :], rt[:, 1:, :]


if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from utils.funcs import *
    from utils.poisson_generated_data import PoissonTimeShiftedData

    # number of populations/hidden units, and number of neurons per population
    n_h = 10  # 3
    neurons_per_population = 20

    # create temporal connections
    U_hat = create_U_hat(n_h) / 0.15

    s = PoissonTimeShiftedData(
        neurons_per_population=20,
        n_populations=n_h,
        n_batches=50,
        norm=1,
        time_steps_per_batch=10000,
        fr_mode='gaussian', frequency_range=[20, 25],
        temporal_connections=U_hat,
    )

    data = reshape(reshape(s.data), T=20, n_batches=2500)
    train, test = data[..., :2000], data[..., 2000:]

    rtrbm = RTRBM(data, n_hidden=8, device="cpu")
    rtrbm.learn(batch_size=10, n_epochs=500, lr=5e-4, CDk=10, mom=0.9, wc=0, sp=0, x=0)
    plt.plot(rtrbm.errors)
    plt.show()

    # Infer from trained RTRBM and plot some results
    vt_infer, rt_infer = rtrbm.infer(torch.tensor(data[:, :10, 0]), t_extra=10)

    # effective coupling
    W = rtrbm.W.detach().clone().numpy()
    U = rtrbm.U.detach().clone().numpy()
    rt = np.array(rtrbm._parallel_recurrent_sample_r_given_v(data))
    data = data.detach().numpy()
    var_h_matrix = np.reshape(np.var(rt[..., 0], 1).repeat(W.shape[1]), [W.shape[1], W.shape[0]]).T
    var_v_matrix = np.reshape(np.var(data[..., 0], 1).repeat(W.shape[0]), [W.shape[0], W.shape[1]])

    Je_Wv = np.matmul(W.T, W * var_h_matrix) / W.shape[1] ** 2
    Je_Wh = np.matmul(W * var_v_matrix, W.T) / W.shape[0] ** 2

    _, ax = plt.subplots(2, 3, figsize=(12, 12))
    sns.heatmap(vt_infer.detach().numpy(), ax=ax[0, 0], cbar=False)
    ax[0, 0].set_title('Infered data')
    ax[0, 0].set_xlabel('Time')
    ax[0, 0].set_ylabel('Neuron index')

    ax[0, 1].plot(rtrbm.errors)
    ax[0, 1].set_title('RMSE of the RTRBM over epoch')
    ax[0, 1].set_xlabel('Epoch')
    ax[0, 1].set_ylabel('RMSE')

    sns.heatmap(Je_Wv, ax=ax[0, 2])
    ax[0, 2].set_title('Effective coupling V')
    ax[0, 2].set_xlabel("Visibel nodes")
    ax[0, 2].set_ylabel("Visibel nodes")

    sns.heatmap(rtrbm.W.detach().numpy(), ax=ax[1, 0])
    ax[1, 0].set_title('Visible to hidden connection')
    ax[1, 0].set_xlabel('Visible')
    ax[1, 0].set_ylabel('Hiddens')

    sns.heatmap(rtrbm.U.detach().numpy(), ax=ax[1, 1])
    ax[1, 1].set_title('Hidden to hidden connection')
    ax[1, 1].set_xlabel('Hidden(t-1)')
    ax[1, 1].set_ylabel('Hiddens(t)')

    sns.heatmap(Je_Wh, ax=ax[1, 2])
    ax[1, 2].set_title('Effective coupling H')
    ax[1, 2].set_xlabel("Hidden nodes [t]")
    ax[1, 2].set_ylabel("Hidden nodes [t]")
    plt.tight_layout()
    plt.show()
