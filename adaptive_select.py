import numpy as np
import torch


def create_adap_data_buffer(window, ts, thres, num_fine_grid, num_layers_adap,
                            buffer_start_steps=2, obs_dim=2, verbose=0, return_index=0):
    """
    create_adap_data_buffer: Dyadic algorithm for selecting adaptive steps to create irregular mesh

    :window: a batch of time series sequences to be processed, could be multi-dimensional. The shape is (bacth_size, window_length, obs_dim).
            The type is torch.tensor
    :ts: the corresponding time stamps. The shape is (bacth_size, window_length). The type is torch.tensor
    :thres: the threshold epsilon that determines which time steps are to be selected. Larger thres leads to fewer steps. thres is a scalar.
    :num_fine_grid: the number of finest grids of the windows
    :num_layers_adap: the number of maximum iterations of the adaptive selection process. num_layers_adap is an integer >= 1.
    :buffer_start_steps: the number of buffer steps to be added before each window. buffer_start_steps is an integer >= 0.
    :obs_dim: the dimension of the data
    :verbose: an integer that determines whether to print the information
    :return_index: an integer that decides whether the remaining indexes are to be kept

    :return: returns the list consisting of data and time stamps after adaptive selection
    """

    train_windows_adap1 = []  # selected data
    train_ts_adap1 = []  # selected time stamps
    train_len_adap1 = []  # remaining length
    time_ind_remain1 = []  # remaining index

    t_ind_list = np.arange(num_fine_grid + 1)
    num_shrink = (window.shape[1] - 1) // num_fine_grid
    # -----------------------------------------------------------------------------------
    for i in range(window.shape[0]):  # process each window separately

        index_temp = np.zeros((ts.shape[1],))

        train_data_temp = window[i, ::num_shrink, :]
        train_ts_temp = ts[i, ::num_shrink]

        time_steps_delete = []  # the set of time stamps to be remove

        for layer in range(num_layers_adap - 1):  # for each layer
            len_temp = 2 ** (layer + 1)
            if layer == 0:
                train_data_temp_diff = np.diff(train_data_temp, axis=0)
                train_data_temp_diff = np.sqrt(np.sum(train_data_temp_diff ** 2, axis=1))
                train_data_temp_diff = train_data_temp_diff / np.diff(train_ts_temp)
                # compute the variation in each finest subinterval (in l_2 norm).
                # the variation can be chosen case by case

            if layer >= 1:
                time_keep_flag_cur = time_keep_flag
                # an indicator of whether each point was kept in the last round

            num_grid_temp = num_fine_grid // 2 ** (layer + 1)
            time_keep_flag = np.ones((num_grid_temp,))
            # an indicator of whether each point will be kept in the new round

            for j in range(num_grid_temp):
                if layer == 0:
                    if max(train_data_temp_diff[len_temp * j:len_temp * (j + 1)]) <= thres:
                        time_keep_flag[j] = 0
                        time_steps_delete.append(len_temp * j + len_temp // 2)
                else:
                    if time_keep_flag_cur[2 * j] + time_keep_flag_cur[2 * j + 1] == 0:
                        train_data_temp_diff1 = np.diff(
                            train_data_temp[len_temp * j:len_temp * (j + 1) + 1:len_temp // 2], axis=0)
                        train_data_temp_diff1 = np.sqrt(np.sum(train_data_temp_diff1 ** 2, axis=1))
                        train_data_temp_diff1 = train_data_temp_diff1 / np.diff(
                            train_ts_temp[len_temp * j:len_temp * (j + 1) + 1:len_temp // 2])

                        if train_data_temp_diff1.max() <= thres:
                            time_keep_flag[j] = 0
                            time_steps_delete.append(len_temp * j + len_temp // 2)

        time_steps_remain = np.delete(t_ind_list, time_steps_delete)  # the indexes to be kept
        train_len_adap1.append(time_steps_remain.shape[0])
        train_ts_adap1.append(train_ts_temp[time_steps_remain])
        train_windows_adap1.append(train_data_temp[time_steps_remain, :])
        if return_index:
            time_ind_remain_temp = time_steps_remain * num_shrink
            index_temp[time_ind_remain_temp] = 1
            time_ind_remain1.append(index_temp)

    # -----------------------------------------------------------------------------------
    # padding, so that the output sequences have the same length
    for i in range(window.shape[0]):
        train_data_temp = train_windows_adap1[i]
        train_ts_temp = train_ts_adap1[i]
        num_temp = max(train_len_adap1) - train_data_temp.shape[0]
        train_ts_temp = torch.cat((train_ts_temp, train_ts_temp[-1].repeat(num_temp)))
        train_data_temp = torch.cat((train_data_temp, train_data_temp[-1, :].reshape((1, obs_dim)).repeat(num_temp, 1)))

        train_windows_adap1[i] = train_data_temp
        train_ts_adap1[i] = train_ts_temp

    train_windows_adap1 = torch.stack(train_windows_adap1)
    train_ts_adap1 = torch.stack(train_ts_adap1)

    # -----------------------------------------------------------------------------------
    # add buffer zone to adaptively selected windows
    train_windows_adap1_buffer = torch.zeros((train_windows_adap1.shape[0],
                                              train_windows_adap1.shape[1] + buffer_start_steps, obs_dim))
    train_ts_adap1_buffer = torch.zeros((train_windows_adap1.shape[0],
                                         train_windows_adap1.shape[1] + buffer_start_steps))
    train_len_adap1_buffer = [0] * train_windows_adap1.shape[0]
    for i in range(train_windows_adap1.shape[0]):
        train_windows_adap1_buffer[i, :, :] = torch.cat(
            (train_windows_adap1[i, 0, :].reshape(1, obs_dim).repeat(buffer_start_steps, 1),
             train_windows_adap1[i, :, :]))
        train_len_adap1_buffer[i] = train_len_adap1[i] + buffer_start_steps

    train_ts_adap1_buffer[:, buffer_start_steps:] = train_ts_adap1
    deltat = ts[0][1] - ts[0][0]
    train_ts_adap1_buffer[:, :buffer_start_steps] = (
            train_ts_adap1[:, 0].reshape(train_ts_adap1.shape[0], 1).repeat(1, buffer_start_steps)
            + torch.linspace(-deltat * buffer_start_steps, -deltat, buffer_start_steps))
    # -----------------------------------------------------------------------------------
    if verbose:
        print('thres = %.2f, averaged window length of adaptive data = %.1f' % (thres, np.mean(train_len_adap1)))

    if return_index:
        return train_windows_adap1_buffer, train_ts_adap1_buffer, train_len_adap1_buffer, np.mean(
            train_len_adap1), np.stack(time_ind_remain1)
    else:
        return train_windows_adap1_buffer, train_ts_adap1_buffer, train_len_adap1_buffer, np.mean(train_len_adap1)
