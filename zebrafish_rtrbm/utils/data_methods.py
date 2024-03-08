import torch
import numpy as np



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
        # if n_conflicts > 0:
        #    print("Found {} conflict(s)".format(n_conflicts))

        # resolve conflicts if there are any
        while n_conflicts > 0:
            # Any conflict in this data set is between two populations getting assigned the same hidden unit.
            # To solve this, the correlation with the hidden unit is checked for both populations;
            # The one with the highest correlation gets to keep its assignment, the other one
            # has to switch to a different hidden unit which has no assignments yet.
            # (If multiple options are still available, it chooses the one with the highest correlation)

            desired_assignment = np.where(occurence_counts == 2)[0][0]  # look at first conflicting assignment
            leftover_assignments = np.where(occurence_counts == 0)[0]  # left over options

            conflicting_populations = np.where(shuffle_idx == desired_assignment)[0]  # will give two results

            loser = np.argmin(corr[
                                  conflicting_populations, desired_assignment])  # 0 or 1. Loser is the population with lower correlation  to desired assignment

            # loser switches its assignment to the best still available option
            new_loser_assignment = leftover_assignments[
                np.argmax(corr[conflicting_populations[loser], leftover_assignments])]
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

