import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from zebrafish_rtrbm.utils.metrics import cross_correlation
from zebrafish_rtrbm.utils.data_methods import create_U_hat


class PoissonTimeShiftedData(object):
    def __init__(
            self, neurons_per_population=10, n_populations=20, n_batches=200, time_steps_per_batch=100, steps_to_settle=0,
             delay=1, U=None, norm=1, seed=None, **kwargs):

        """
        """
        if 'int_range' not in kwargs:  # Interval Range External Input
            kwargs['int_range'] = [5, 10]
        if 'max_fr' not in kwargs:  # upper bound of firing rate of external input
            kwargs['max_fr'] = 1
        if 'std_range' not in kwargs:  # width of gaussian shaped peaks
            kwargs['std_range'] = [1, 2]
        if 'bound_fr_range' not in kwargs:  # lower and upper bound of firing rate after modulation
            kwargs['bound_fr_range'] = [0, 3*kwargs['max_fr']]
        if 'spread_fr' not in kwargs: # firing rate spread inside one hidden population
            kwargs['spread_fr'] = [0.6, 1.4]

        self.kwargs = kwargs
        
        if seed is not None:
            np.random.seed(seed)
            torch.random.manual_seed(seed)

        # If U is not given, create U with sequential firing of populations
        if U is None:
            U = create_U_hat(n_populations)
            U /= norm

        self.data = torch.empty(
            neurons_per_population * n_populations,
            time_steps_per_batch,
            n_batches,
            dtype=torch.float
        )

        time_steps_to_simulate = time_steps_per_batch + steps_to_settle + delay
        population_waves = torch.zeros(n_populations, time_steps_to_simulate, n_batches)
        neuron_waves_interact = torch.zeros(neurons_per_population * n_populations, time_steps_per_batch, n_batches)
        idx = [torch.randperm(neurons_per_population) for _ in range(n_populations)]

        # get all mother trains by looping over populations
        for h in range(n_populations):
            background_wave = self.get_random_peaks(
                time_steps_per_batch=time_steps_to_simulate, n_batches=n_batches,
                **kwargs) / 10

            population_waves[h, :, :] = self.get_random_peaks(
                time_steps_per_batch=time_steps_to_simulate, n_batches=n_batches,
                **kwargs) + background_wave

        self.population_waves_original = population_waves.detach().clone()

        # compute interactions of all populations on their resulting firing rate
        for t in range(delay, time_steps_to_simulate):
            for h in range(n_populations):
                population_waves[h, t, :] += torch.sum(
                    U[h, :][None, :, None] * population_waves[:, t - delay, :], (0, 1)
                )

            # constrain to only positive values, lower & upper limit and remove nan
            population_waves[:, t, :] = self.constraints(population_waves[:, t, :], **kwargs)

        # cut first temporal part
        population_waves = population_waves[:, (delay+steps_to_settle):, :]

        for h in range(n_populations):
            neuron_waves_interact[neurons_per_population * h: neurons_per_population * (h + 1), ...] = \
                (population_waves[h, ...]).repeat(neurons_per_population, 1, 1) * \
                torch.linspace(kwargs['spread_fr'][0], kwargs['spread_fr'][1], neurons_per_population)[
                    idx[h], None, None]

        self.data = torch.poisson(neuron_waves_interact)

        # make sure there are
        self.data[self.data < 0] = 0
        self.data[self.data > 1] = 1
        self.population_waves_interact = population_waves
        self.neuron_waves_interact = neuron_waves_interact
        self.firing_rates = torch.mean(self.data, (1, 2))
        self.delay = delay
        self.time_steps_per_batch = time_steps_per_batch
        self.U = U

    def gaussian_pdf(self, x, mu, std):
        pdf = 1 / (torch.sqrt(torch.tensor(np.pi)) * std) * torch.exp(-0.5 * ((x - mu) / std) ** 2)
        return pdf

    def get_random_peaks(self, time_steps_per_batch, n_batches, **kwargs):
        T = torch.arange(time_steps_per_batch)

        #return kwargs['max_fr'] * torch.normal(0.05, 0.01, size=(time_steps_per_batch, n_batches))
        # get peak locations of the external input
        isi = torch.randint(low=kwargs['int_range'][0], high=kwargs['int_range'][1],
                            size=(time_steps_per_batch // kwargs['int_range'][0], n_batches))
        trace = torch.empty(size=(time_steps_per_batch, n_batches))

        for batch in range(n_batches):
            mu = torch.cumsum(isi[:, batch], dim=0)
            mu -= torch.randint(low=0, high=kwargs['int_range'][0], size=(1,)) # random starting point
            mu = mu[mu < time_steps_per_batch]
            n_samples = mu.shape[0]
            trace[:, batch] = torch.sum(self.gaussian_pdf(T[:, None], mu,
                                                          torch.rand(1, n_samples) * (
                                                                      kwargs['std_range'][1] - kwargs['std_range'][0]) +
                                                          kwargs['std_range'][0]), 1)
        # normalisation such that max possible peak is at 1
        trace *= torch.sqrt(torch.tensor(np.pi)) * kwargs['std_range'][0]
        return kwargs['max_fr'] * trace # modulated trace based on desired peak firing rate

    def constraints(self, population_waves_interact, **kwargs):
        population_waves_interact[population_waves_interact < abs(kwargs['bound_fr_range'][0])] = abs(
            kwargs['bound_fr_range'][0])
        population_waves_interact[population_waves_interact > kwargs['bound_fr_range'][1]] = kwargs['bound_fr_range'][1]
        population_waves_interact[torch.isnan(population_waves_interact)] = kwargs['bound_fr_range'][0]
        return population_waves_interact

    def plot_stats(self, T=None, batch=0, axes=None, t_cross_corr=12):
        if T is None or T > self.time_steps_per_batch:
            T = self.time_steps_per_batch

        if axes is None:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        sns.heatmap(self.data[:, :T, batch], ax=axes[0, 0], cbar=False)
        axes[0, 0].set_title('Final spikes')

        axes[0, 1].plot(self.firing_rates[torch.argsort(self.firing_rates)], '.')
        axes[0, 1].set_title('Mean firing rates (over all batches)')

        sns.heatmap(self.U, ax=axes[0, 2], cbar=False)
        axes[0, 2].set_title('Hidden population structure')
        maxi = torch.tensor(0)
        for i, (wave_O, wave_I) in enumerate(
                zip(self.population_waves_original[..., batch], self.population_waves_interact[..., batch])):
            axes[1, 0].plot(wave_O[:T], label=str(i))
            axes[1, 1].plot(wave_I[:T], label=str(i))
            maxi = torch.max(maxi, torch.max(torch.cat([wave_O[:T], wave_I[:T]])))

        axes[1, 0].set_title('Original population waves')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_xticks([0, T + self.delay])
        axes[1, 0].set_xticklabels(['0', str(T + self.delay)])
        axes[1, 0].set_ylim([0, maxi])

        axes[1, 1].set_title('Population waves after interaction')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_xticks([0, T])
        axes[1, 1].set_xticklabels([str(self.delay), str(T + self.delay)])
        axes[1, 1].set_ylim([0, maxi])

        cross_corr = torch.zeros(t_cross_corr)
        for i in range(t_cross_corr):
            cross_corr[i] = np.nanmean(cross_correlation(data=self.data[:, :, 0], time_shift=i, mode='Pearson'))
        axes[1, 2].plot(cross_corr)
        axes[1, 2].set_title('Pearson cross-correlation')
        axes[1, 2].set_xlabel('Time shift')
        axes[1, 2].set_ylabel('Cross-correlation')

        return axes
