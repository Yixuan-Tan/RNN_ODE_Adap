import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep, interp1d
from datetime import timedelta
from interpolate_windows import interpolate_window_adap, interpolate_window
from NN import RNNODE, OutputNN


def fit_adap_models(odefunc, outputfunc, window_orig, ts_orig, window_adap_buffer, ts_adap_buffer, len_adap_buffer,
                    buffer_start_steps, interp_kind='linear', rescale_const=1., verbose=1, method='NaiveEuler',
                    obs_dim=2, n_latent=128, time_scale=10):
    # 1-step prediction results on training data (interpolation)
    input_len = window_adap_buffer.shape[1]

    #     print(window_orig.shape[0], window_orig.shape[0] > 1)
    if window_orig.shape[0] > 1 and method != 'NaiveEuler':
        print('adaptive method for more than 1 windows cannot use numerical methods except for forward Euler.')
        return False

    with torch.no_grad():
        h0 = torch.zeros(window_adap_buffer.shape[0], n_latent)
        xhat_window_adap_buffer_train1_buffer = []
        xhat_window_adap_buffer_train1_buffer.append(window_adap_buffer[:, 0, :].numpy())
        for t_ind in range(input_len - 1):
            x_temp = window_adap_buffer[:, t_ind, :] * rescale_const
            deltat_mult = (ts_adap_buffer[:, t_ind + 1] - ts_adap_buffer[:, t_ind])
            if method == 'NaiveEuler':
                h1 = h0 + odefunc(t=ts_adap_buffer[0, 0], x=x_temp, h=h0) * deltat_mult.reshape(
                    window_adap_buffer.shape[0], 1).repeat(1, n_latent) / time_scale
            # else:
            #     h1 = odeint(lambda t, h: odefunc(t=t, x=x_temp, h=h),
            #                 h0, ts_adap_buffer[0][t_ind:t_ind + 2] / time_scale, method=method)
            #     h1 = h1[-1, :]

            xhat = outputfunc(h1) / rescale_const
            xhat_window_adap_buffer_train1_buffer.append(xhat.detach().numpy())
            h0 = h1

    xhat_window_adap_buffer_train1_buffer = np.transpose(np.array(xhat_window_adap_buffer_train1_buffer), (1, 0, 2))

    xhat_window_temp = np.zeros((xhat_window_adap_buffer_train1_buffer.shape[0], window_orig.shape[1] - 1, obs_dim))
    for i in range(xhat_window_adap_buffer_train1_buffer.shape[0]):
        xk = ts_adap_buffer[i][buffer_start_steps:len_adap_buffer[i]]
        x2 = ts_orig[i]
        #         print(xk, x2)
        for k in range(obs_dim):
            yk = xhat_window_adap_buffer_train1_buffer[i, buffer_start_steps:len_adap_buffer[i], k]
            spl = interp1d(xk, yk, kind=interp_kind)
            xhat_window_temp[i, :, k] = torch.tensor(spl(x2))[1:]
    xhat_window_adap_buffer_train1_buffer1 = xhat_window_temp

    # compute errors

    fit_L2_err_adap_train1 = torch.mean(torch.sum((window_orig[:, 1:, :] - xhat_window_adap_buffer_train1_buffer1) ** 2
                                                  , dim=(2)),
                                        dim=1) ** 0.5  # /xhat_window_adap_buffer_train1_buffer.shape[1]
    fit_L1_err_adap_train1 = torch.mean(torch.sum((window_orig[:, 1:, :] - xhat_window_adap_buffer_train1_buffer1) ** 2
                                                  , dim=(2)) ** 0.5, dim=1)
    if verbose:
        print('L1 err = %.3e \pm %.3e, L2 err = %.3e \pm %.3e' %
              (fit_L1_err_adap_train1.mean(), fit_L1_err_adap_train1.std(),
               fit_L2_err_adap_train1.mean(), fit_L2_err_adap_train1.std()))
    return xhat_window_adap_buffer_train1_buffer1, fit_L2_err_adap_train1, fit_L1_err_adap_train1


def pred_adap_models(odefunc, outputfunc, window_orig, ts_orig, window_adap_buffer_trunc1, ts_adap_buffer_trunc1,
                      ts_adap_buffer_trunc2, len_adap_buffer_trunc2, pred_len, buffer_start_steps=2, verbose=1,
                      interp_kind='linear', rescale_const=1., obs_dim=2, n_latent=128, time_scale=10):
    # ts_adap_buffer_trunc2 needs to include the starting point of the second half
    # (i.e. the middle point of the whole interval)

    input_len_buffer = ts_adap_buffer_trunc1.shape[1]
    pred_len_buffer = ts_adap_buffer_trunc2.shape[1]
    pred_len = (ts_orig.shape[1] - 1) // 2

    with torch.no_grad():
        h0 = torch.zeros(window_adap_buffer_trunc1.shape[0], n_latent)
        pred_window_adap1 = []
        xhat_window_adap1 = []

        for t_ind in range(input_len_buffer - 1):
            x_temp = window_adap_buffer_trunc1[:, t_ind, :] * rescale_const
            deltat_mult = (ts_adap_buffer_trunc1[:, t_ind + 1] - ts_adap_buffer_trunc1[:, t_ind])
            h1 = h0 + odefunc(t=ts_adap_buffer_trunc1[0, 0], x=x_temp, h=h0) * deltat_mult.reshape(
                window_adap_buffer_trunc1.shape[0], 1).repeat(1, n_latent) / time_scale
            xhat = outputfunc(h1) / rescale_const
            xhat_window_adap1.append(xhat.detach().numpy())
            h0 = h1

        pred_window_adap1.append(window_adap_buffer_trunc1[:, -1, :].numpy())
        for t_ind in range(pred_len_buffer - 1):
            x_temp = xhat.detach() * rescale_const
            #             if t_ind == 0:
            #                 x_temp = window_adap_buffer_trunc1[:,-1,:] * rescale_const
            #             else:
            #                 x_temp = xhat.detach() * rescale_const
            deltat_mult = (ts_adap_buffer_trunc2[:, t_ind + 1] - ts_adap_buffer_trunc2[:, t_ind])
            h1 = h0 + odefunc(t=ts_adap_buffer_trunc1[0, 0], x=x_temp, h=h0) * deltat_mult.reshape(
                window_adap_buffer_trunc1.shape[0], 1).repeat(1, n_latent) / time_scale
            xhat = outputfunc(h1) / rescale_const
            pred_window_adap1.append(xhat.detach().numpy())
            h0 = h1

    pred_window_adap1 = np.transpose(np.array(pred_window_adap1), (1, 0, 2))
    xhat_window_adap1 = np.transpose(np.array(xhat_window_adap1), (1, 0, 2))

    #     print(pred_window_adap1[0])

    pred_window_adap1_temp = np.zeros((pred_window_adap1.shape[0], (ts_orig.shape[1] - 1) // 2, obs_dim))
    for i in range(window_orig.shape[0]):
        xk = ts_adap_buffer_trunc2[i, :len_adap_buffer_trunc2[i]]
        x2 = ts_orig[i, (ts_orig.shape[1] + 1) // 2:]
        #         print(xk, x2)
        for k in range(pred_window_adap1.shape[2]):
            yk = pred_window_adap1[i, :len_adap_buffer_trunc2[i], k]
            #             print(xk.shape, yk.shape)
            f = interp1d(xk, yk, kind=interp_kind)
            pred_window_adap1_temp[i, :, k] = torch.tensor(f(x2))  # [1:]
    pred_window_adap1 = pred_window_adap1_temp

    pred_L2_err_adap_train1 = torch.mean(torch.sum((window_orig[:, -pred_len:, :] - pred_window_adap1) ** 2
                                                   , dim=(2)), dim=1) ** 0.5  # /pred_window_adap1.shape[1]
    pred_L1_err_adap_train1 = torch.mean(torch.sum((window_orig[:, -pred_len:, :] - pred_window_adap1) ** 2
                                                   , dim=(2)) ** 0.5, dim=1)  # /pred_window_adap1.shape[1]

    if verbose:
        print('L1 err = %.3e \pm %.3e, L2 err = %.3e \pm %.3e' %
              (pred_L1_err_adap_train1.mean(), pred_L1_err_adap_train1.std(),
               pred_L2_err_adap_train1.mean(), pred_L2_err_adap_train1.std()))
    return xhat_window_adap1, pred_window_adap1, pred_L2_err_adap_train1, pred_L1_err_adap_train1


def train_adap_models(train_dataloader, input_steps, verbose, valid_window, valid_ts, valid_len, valid_window_adap,
                       valid_ts_adap, valid_window_adap_trunc, valid_ts_adap_trunc, valid_ts_adap_trunc2,
                       valid_len_adap_trunc2, pred_len, num_train_windows=500,
                       lr=1e-2, n_iter=400, n_latent=128, n_hidden=128, obs_dim=2, rescale_const=1.,
                       thres1=3.5e2, thres2=5e1, buffer_start_steps=2, weight=(1, 0), interp_kind='linear', time_scale=10):
    odefunc = RNNODE(input_dim=obs_dim, n_latent=n_latent, n_hidden=n_hidden)
    outputfunc = OutputNN(input_dim=obs_dim, n_latent=n_latent)
    params_adap1_buffer = list(odefunc.parameters()) + list(outputfunc.parameters())
    opt_adap1_buffer = optim.Adam(params_adap1_buffer, lr=lr)
    train_loss_adap1_buffer = []
    fit_err = []
    min_valid_loss = 1e5

    t1 = time.time()
    for itr in range(1, n_iter + 1):
        # for itr in range(1,11):    
        loss_epoch = 0.
        for batch_idx, samples in enumerate(train_dataloader):

            samples, train_ts_temp, samples_len = samples[0], samples[1], samples[2]
            h0 = torch.zeros(samples.shape[0], n_latent)
            window_ind_list = np.arange(samples.shape[0])
            loss = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
            opt_adap1_buffer.zero_grad()
            for t_ind in range(input_steps - 1):

                window_ind_temp = window_ind_list[samples_len >= t_ind + 2]
                if window_ind_temp.size > 0:
                    x_temp = samples[window_ind_temp, t_ind, :] * rescale_const
                    deltat_mult = (train_ts_temp[window_ind_temp, t_ind + 1] - train_ts_temp[window_ind_temp, t_ind])
                    h1 = h0[window_ind_temp, :] + odefunc(t=train_ts_temp[0, 0], x=x_temp, h=h0[window_ind_temp, :]
                                                          ) * deltat_mult.reshape(window_ind_temp.size, 1).repeat(1,
                                                                                                                  n_latent) / time_scale
                    xhat = outputfunc(h1) / rescale_const
                    loss = loss + torch.sum((xhat - samples[window_ind_temp, t_ind + 1, :]) ** 2
                                            * deltat_mult.reshape(window_ind_temp.size, 1).repeat(1,
                                                                                                  obs_dim)) / time_scale
                    h0[window_ind_temp, :] = h1

            loss_epoch += loss.item()
            loss.backward()
            opt_adap1_buffer.step()

        _, _, valid_loss1, _ = pred_adap_models(odefunc, outputfunc, valid_window, valid_ts, valid_window_adap_trunc,
                                                 valid_ts_adap_trunc, valid_ts_adap_trunc2,
                                                 valid_len_adap_trunc2, buffer_start_steps=buffer_start_steps,
                                                 verbose=0, interp_kind=interp_kind, pred_len=pred_len,
                                                 rescale_const=rescale_const,
                                                 obs_dim=obs_dim, n_latent=n_latent, time_scale=time_scale)
        _, valid_loss2, _ = fit_adap_models(odefunc, outputfunc, valid_window, valid_ts, valid_window_adap,
                                            valid_ts_adap,
                                            buffer_start_steps=buffer_start_steps, verbose=0, interp_kind=interp_kind,
                                            len_adap_buffer=valid_len, rescale_const=rescale_const,
                                            obs_dim=obs_dim, n_latent=n_latent, time_scale=time_scale)
        valid_loss1, valid_loss2 = valid_loss1.mean(), valid_loss2.mean()
        valid_loss = weight[0] * valid_loss1 + weight[1] * valid_loss2
        fit_err.append(valid_loss)
        if valid_loss <= min_valid_loss:
            torch.save([odefunc, outputfunc], 'saved_model8.pth')
            min_valid_loss = valid_loss

        t2 = time.time()
        remain_time = (t2 - t1) / itr * (n_iter - itr)
        train_loss_adap1_buffer.append(loss_epoch / num_train_windows)
        if verbose >= 2:
            print('the %d-th epoch, training loss = %.4e' % (itr, loss_epoch / num_train_windows)
                  + ', remaining time = ' + str(timedelta(seconds=remain_time)))
        elif verbose == 1 and (itr == n_iter or itr == 1):
            print('the %d-th epoch, training loss = %.4e' % (itr, loss_epoch / num_train_windows)
                  + ', remaining time = ' + str(timedelta(seconds=remain_time)))
        if loss_epoch / num_train_windows >= thres1 or (
                itr >= 10 and loss_epoch / num_train_windows >= thres2):
            return False, odefunc, outputfunc

    t3 = time.time()
    plt.plot(train_loss_adap1_buffer)
    plt.show()

    plt.plot(fit_err)
    plt.title('validation error')
    [odefunc, outputfunc] = torch.load('saved_model8.pth')
    _, _, valid_loss1, _ = pred_adap_models(odefunc, outputfunc, valid_window, valid_ts, valid_window_adap_trunc,
                                             valid_ts_adap_trunc, valid_ts_adap_trunc2,
                                             valid_len_adap_trunc2, buffer_start_steps=buffer_start_steps, verbose=0,
                                             interp_kind=interp_kind, pred_len=pred_len, rescale_const=rescale_const,
                                             obs_dim=obs_dim, n_latent=n_latent, time_scale=time_scale)
    _, valid_loss2, _ = fit_adap_models(odefunc, outputfunc, valid_window, valid_ts, valid_window_adap, valid_ts_adap,
                                        buffer_start_steps=buffer_start_steps, verbose=0, interp_kind=interp_kind,
                                        len_adap_buffer=valid_len, rescale_const=rescale_const,
                                        obs_dim=obs_dim, n_latent=n_latent, time_scale=time_scale)
    valid_loss1, valid_loss2 = valid_loss1.mean(), valid_loss2.mean()
    valid_loss = weight[0] * valid_loss1 + weight[1] * valid_loss2
    plt.axhline(valid_loss, color='red', linestyle='--')
    plt.show()
    print('Training Neural ODE takes %.2f s' % (t3 - t1))

    return True, odefunc, outputfunc
