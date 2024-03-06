"""
Implementation of the Restricted Boltzmann Machine (RBM)
"""

import torch
import numpy as np
from tqdm import tqdm
from utils.lr_scheduler import get_lrs
from utils.funcs import reshape


class RBM(object):

    def __init__(self, data, n_hidden=10, device=None, debug_mode=False, save_every_n_epochs=1):

        if device is None:
            self.device = "cpu" if not torch.cuda.is_available() else "cuda:0"
        else:
            self.device = device

        self.dtype = torch.float
        self.data = data.float().to(self.device)
        self.dim = torch.tensor(self.data.shape).shape[0]

        if self.dim == 1:
            self.n_visible = self.data.shape
        elif self.dim == 2:
            self.n_visible, self.num_samples = self.data.shape
        elif self.dim == 3:
            self.data = reshape(data).float().to(self.device)
            self.n_visible, self.num_samples = self.data.shape
            self.dim = torch.tensor(self.data.shape).shape[0]
        else:
            raise ValueError("Data is not correctly defined: Use (n_visible) or (n_visible, num_samples) dimensions.\
                             If you want to have (n_visible, T, num_samples) try to reshape it to (n_visible, T*num_samples).\
                             And if you want to train on each sample separately set batchsize=T.")

        self.n_hidden = n_hidden

        self.W = 0.01 * torch.randn(self.n_hidden, self.n_visible, dtype=self.dtype, device=self.device)
        self.b_H = torch.zeros(self.n_hidden, dtype=self.dtype, device=self.device)
        self.b_V = torch.zeros(self.n_visible, dtype=self.dtype, device=self.device)

        self.params = [self.W, self.b_H, self.b_V]
        self.dparams = self.initialize_grad_updates()
        self.errors = []

        if debug_mode:
            self.parameter_history = []

        self.save_every_n_epochs = save_every_n_epochs
        self.debug_mode= debug_mode

    def learn(self,
              n_epochs=1000,
              batchsize=1,
              CDk=10, PCD=False,
              lr=1e-3, lr_regulisor=1, lr_schedule=None,
              sp=None, x=2,
              mom=0.9,
              wc=0.0002,
              AF=torch.sigmoid,
              disable_tqdm=False, reshuffle_batch=True, **kwargs):

        global vt_k
        if self.dim == 1:
            num_batches = 1
            batchsize = 1
        elif self.dim == 2:
            num_batches = self.num_samples // batchsize

        if lr is None:
            lrs = lr_regulisor * np.array(get_lrs(mode=lr_schedule, n_epochs=n_epochs, **kwargs))
        else:
            lrs = lr * torch.ones(n_epochs)

        Dparams = self.initialize_grad_updates()
        self.disable = disable_tqdm

        for epoch in tqdm(range(0, n_epochs), disable=self.disable):
            err = 0

            for batch in range(num_batches):
                self.dparams = self.initialize_grad_updates()

                v = self.data[:, batch * batchsize : (batch+1) * batchsize].to(self.device)

                # Perform contrastive divergence and compute model statistics
                if PCD and epoch != 0:
                    # use last gibbs sample as input (Persistant Contrastive Divergence)
                    vk, pvk, hk, phk, ph, h = self.CD(vk, CDk, AF=AF)
                    ph, h = self.visible_to_hidden(v, AF=AF)
                else:
                    # use data (normal Contrastive Divergence)
                    vk, pvk, hk, phk, ph, h = self.CD(v, CDk, AF=AF)

                # Accumulate error
                err += torch.sum((v - vk) ** 2).cpu()

                # Compute gradients
                self.dparams = self.grad(v, vk, ph, phk)

                # Update gradients
                Dparams = self.update_grad(Dparams, lr=lrs[epoch], mom=mom, wc=wc, sp=sp, x=x)

            self.errors += [err / self.data.numel()]

            if self.debug_mode and epoch % self.save_every_n_epochs == 0:
                self.parameter_history.append([param.detach().clone().cpu() for param in self.params])

            if reshuffle_batch:
                self.data[..., :] = self.data[..., torch.randperm(self.num_samples)]

    def CD(self, v, CDk, AF=torch.sigmoid):

        ph, h = self.visible_to_hidden(v)
        hk = h.detach().clone()
        for k in range(CDk):
            pvk, vk = self.hidden_to_visible(hk, AF=AF)
            phk, hk = self.visible_to_hidden(vk, AF=AF)

        return vk, pvk, hk, phk, ph, h

    def visible_to_hidden(self, v, AF=torch.sigmoid):
        p = torch.sigmoid(torch.einsum('hv, vb->hb', self.W, v) + self.b_H[:, None])
        return p, torch.bernoulli(p)

    def hidden_to_visible(self, h, AF=torch.sigmoid):
        p = torch.sigmoid(torch.einsum('hv, hb->vb', self.W, h) + self.b_V[:, None])
        return p, torch.bernoulli(p)

    def initialize_grad_updates(self):
        return [torch.zeros_like(param, dtype=self.dtype, device=self.device) for param in self.params]

    def grad(self, v, vk, ph, phk):

        dW = torch.mean(torch.einsum('hb, vb -> hvb', ph, v) - torch.einsum('hb, vb -> hvb', phk, vk), 2)
        db_V = torch.mean(v - vk, 1)
        db_H = torch.mean(ph - phk, 1)

        return [dW, db_H, db_V]

    def update_grad(self, Dparams, lr=1e-3, mom=0, wc=0, sp=None, x=2):

        dW, db_H, db_V = self.dparams
        DW, Db_H, Db_V = Dparams

        DW = mom * DW + lr * (dW - wc * self.W)
        Db_H = mom * Db_H + lr * db_H
        Db_V = mom * Db_V + lr * db_V

        if sp is not None:
            DW -= sp * torch.reshape(torch.sum(torch.abs(self.W), 1).repeat(self.n_visible),
                                     [self.n_hidden, self.n_visible]) ** (x - 1) * torch.sign(self.W)

        Dparams = [DW, Db_H, Db_V]

        for i in range(len(self.params)): self.params[i] += Dparams[i]

        return Dparams

    def free_energy(self, v):

        v_term = torch.outer(v, self.b_V.T)
        w_x_h = torch.nn.functional.linear(v, self.W.T, self.b_H)
        h_term = torch.sum(torch.nn.functional.softplus(w_x_h))

        return torch.mean(-h_term - v_term)

    def sample(self,
               v_start,
               AF=torch.sigmoid,
               pre_gibbs_k=100,
               gibbs_k=20,
               mode=1,
               chain=50,
               disable_tqdm=False):

        if v_start.ndim==1: v_start=v_start[:, None]
        num_samples = v_start.shape[1]
        vt = torch.zeros(self.n_visible, chain+1, num_samples, dtype=self.dtype, device=self.device)
        ht = torch.zeros(self.n_hidden, chain+1, num_samples, dtype=self.dtype, device=self.device)

        v = v_start

        for kk in range(pre_gibbs_k):
            _, h = self.visible_to_hidden(v, AF=AF)
            _, v = self.hidden_to_visible(h, AF=AF)

        for t in tqdm(range(chain + 1), disable=disable_tqdm):
            vt_k = torch.zeros(self.n_visible, num_samples, gibbs_k, dtype=self.dtype, device=self.device)
            ht_k = torch.zeros(self.n_hidden, num_samples, gibbs_k, dtype=self.dtype, device=self.device)
            for kk in range(gibbs_k):
                _, h = self.visible_to_hidden(v, AF=AF)
                _, v = self.hidden_to_visible(h, AF=AF)
                vt_k[..., kk] = v
                ht_k[..., kk] = h

            if mode == 1:
                vt[:, t, :] = vt_k[..., -1]
                ht[:, t, :] = ht_k[..., -1]
            if mode == 2:
                vt[:, t, :] = torch.mean(vt_k, -1)
        vt = reshape(vt[:, 1:, :])
        ht = reshape(ht[:, 1:, :])
        return vt, ht

