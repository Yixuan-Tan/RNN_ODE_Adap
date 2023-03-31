import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep, interp1d
from datetime import timedelta
from interpolate_windows import interpolate_window
from NN import RNNODE, OutputNN
# from torchdiffeq import odeint

def fit_non_adap_models_grids(odefunc, outputfunc, window_orig, ts_orig, num_grids,
                              interp_kind='cubic', method="naiveEuler", buffer_start_steps=2, verbose=1,
                              rescale_const=1., obs_dim=2, n_latent=128, time_scale=10):
    test_ts2 = torch.tensor(np.linspace(ts_orig[:, 0], ts_orig[:, -1], num_grids).T)
    test_ts2 = torch.concat((test_ts2[:, 0].reshape(test_ts2.shape[0], 1).repeat(1, buffer_start_steps),
                             test_ts2), dim=1)

    test_windows2 = interpolate_window(window_orig, ts_orig, test_ts2, 'cubic')

    if buffer_start_steps > 0:
        deltat = test_ts2[0][-1] - test_ts2[0][-2]
        test_ts2[:, :buffer_start_steps] = (test_ts2[:, :buffer_start_steps] +
                                            torch.arange(-deltat * buffer_start_steps, 0, deltat))

    input_len = test_windows2.shape[1]

    with torch.no_grad():
        h0 = torch.zeros(window_orig.shape[0], n_latent)
        xhat_window_test2 = []
        xhat_window_test2.append(test_windows2[:, 0, :].numpy())
        for t_ind in range(input_len - 1):
            x_temp = test_windows2[:, t_ind, :] * rescale_const
            deltat_mult = (test_ts2[:, t_ind + 1] - test_ts2[:, t_ind])
            if method == "naiveEuler":
                h1 = h0 + odefunc(t=ts_orig[0, 0], x=x_temp, h=h0) * deltat_mult.reshape(window_orig.shape[0],
                                                                                         1).repeat(1,
                                                                                                   n_latent) / time_scale
            # else:
            #     h1 = odeint(lambda t, h: odefunc(t=t, x=x_temp, h=h),
            #                 h0, test_ts2[0][t_ind:t_ind + 2] / time_scale, method=method)
            #     h1 = h1[-1, :]
            xhat = outputfunc(h1) / rescale_const
            xhat_window_test2.append(xhat.detach().numpy())
            h0 = h1

    xhat_window_test2 = np.transpose(np.array(xhat_window_test2), (1, 0, 2))

    xhat_window_test2_temp = np.zeros((xhat_window_test2.shape[0], window_orig.shape[1] - 1, obs_dim))
    for i in range(window_orig.shape[0]):
        xk = test_ts2[i, :]
        x2 = ts_orig[i, :]
        #         print(xk)
        #         print(x2)
        for k in range(obs_dim):
            yk = xhat_window_test2[i, :, k]
            f = interp1d(xk, yk, kind=interp_kind)
            xhat_window_test2_temp[i, :, k] = torch.tensor(f(x2))[1:]
    xhat_window_test2 = xhat_window_test2_temp

    # compute errors

    fit_L2_err_test2 = torch.mean(torch.sum((window_orig[:, 1:, :] - xhat_window_test2) ** 2
                                            , dim=(2)), dim=1) ** 0.5  # /xhat_window_test2.shape[1]
    fit_L1_err_test2 = torch.mean(torch.sum((window_orig[:, 1:, :] - xhat_window_test2) ** 2
                                            , dim=(2)) ** 0.5, dim=1)  # /xhat_window_test2.shape[1]

    if verbose:
        print('L1 err = %.3e \pm %.3e, L2 err = %.3e \pm %.3e' %
              (fit_L1_err_test2.mean(), fit_L1_err_test2.std(),
               fit_L2_err_test2.mean(), fit_L2_err_test2.std()))

    return xhat_window_test2, fit_L2_err_test2, fit_L1_err_test2

def pred_non_adap_models_grids(odefunc, outputfunc, window_orig, ts_orig, num_grids, time_scale=10,
                               interp_kind='cubic', buffer_start_steps=2, verbose=1, rescale_const=1., obs_dim=2, n_latent=128):
    test_ts2 = torch.tensor(np.linspace(ts_orig[:, 0], ts_orig[:, -1], num_grids).T)
    test_ts2 = torch.concat((test_ts2[:, 0].reshape(test_ts2.shape[0], 1).repeat(1, buffer_start_steps),
                             test_ts2), dim=1)

    test_windows2 = interpolate_window(window_orig, ts_orig, test_ts2, 'cubic')

    if buffer_start_steps > 0:
        deltat = test_ts2[0][-1] - test_ts2[0][-2]
        test_ts2[:, :buffer_start_steps] = (test_ts2[:, :buffer_start_steps] +
                                            torch.arange(-deltat * buffer_start_steps, 0, deltat))

    input_len = (num_grids + 1) // 2 + buffer_start_steps
    pred_len = (num_grids - 1) // 2

    with torch.no_grad():
        h0 = torch.zeros(window_orig.shape[0], n_latent)
        xhat_window_test2 = []
        pred_window_test2 = []
        xhat_window_test2.append(test_windows2[:, 0, :].numpy())
        for t_ind in range(input_len - 1):
            x_temp = test_windows2[:, t_ind, :] * rescale_const
            deltat_mult = (test_ts2[:, t_ind + 1] - test_ts2[:, t_ind])
            h1 = h0 + odefunc(t=ts_orig[0, 0], x=x_temp, h=h0) * deltat_mult.reshape(window_orig.shape[0], 1).repeat(1,
                                                                                                                     n_latent) / time_scale
            xhat = outputfunc(h1) / rescale_const
            xhat_window_test2.append(xhat.detach().numpy())
            h0 = h1

        pred_window_test2.append(test_windows2[:, t_ind, :].numpy())
        for t_ind in range((input_len - 1), (input_len - 1) + pred_len):
            if t_ind == input_len - 1:
                x_temp = test_windows2[:, t_ind, :] * rescale_const
            else:
                x_temp = xhat.detach() * rescale_const
            deltat_mult = (test_ts2[:, t_ind + 1] - test_ts2[:, t_ind])
            h1 = h0 + odefunc(t=ts_orig[0, 0], x=x_temp, h=h0) * deltat_mult.reshape(window_orig.shape[0], 1).repeat(1,
                                                                                                                     n_latent) / time_scale
            xhat = outputfunc(h1) / rescale_const
            #             if t_ind == input_len-1:
            #                 print(deltat_mult[0], h0[0], h1[0], xhat[0])
            pred_window_test2.append(xhat.detach().numpy())
            h0 = h1

    xhat_window_test2 = np.transpose(np.array(xhat_window_test2), (1, 0, 2))[:, buffer_start_steps:, :]
    pred_window_test2 = np.transpose(np.array(pred_window_test2), (1, 0, 2))
    #     print(xhat_window_test2[0])
    #     print(pred_window_test2[0])
    #     print(xhat_window_test2.shape, pred_window_test2.shape)

    pred_window_test2_temp = np.zeros((pred_window_test2.shape[0], (ts_orig.shape[1] - 1) // 2, obs_dim))
    for i in range(window_orig.shape[0]):
        xk = test_ts2[i, input_len - 1:]
        x2 = ts_orig[i, (ts_orig.shape[1] + 1) // 2:]
        #         print(xk, x2)
        for k in range(pred_window_test2.shape[2]):
            yk = pred_window_test2[i, :, k]
            f = interp1d(xk, yk, kind=interp_kind)
            pred_window_test2_temp[i, :, k] = torch.tensor(f(x2))  # [1:]
    #             print(yk, pred_window_test2_temp[i,:,k])
    pred_window_test2 = pred_window_test2_temp

    # compute errors

    pred_L2_err_test2 = torch.mean(torch.sum((window_orig[:, -(ts_orig.shape[1] - 1) // 2:, :] - pred_window_test2) ** 2
                                             , dim=(2)), dim=1) ** 0.5
    pred_L1_err_test2 = torch.mean(torch.sum(abs(window_orig[:, -(ts_orig.shape[1] - 1) // 2:, :] - pred_window_test2)
                                             , dim=(2)), dim=1)
    if verbose:
        print('L1 err = %.3e \pm %.3e, L2 err = %.3e \pm %.3e' %
              (pred_L1_err_test2.mean(), pred_L1_err_test2.std(),
               pred_L2_err_test2.mean(), pred_L2_err_test2.std()))

    return xhat_window_test2, pred_window_test2, pred_L2_err_test2, pred_L1_err_test2


def train_non_adap_models(train_dataloader, input_steps, valid_window, valid_ts, num_grid,
                           verbose, lr=1e-2, n_iter=400, n_latent=128, num_train_windows=500,
                           n_hidden=128, obs_dim=2, thres1=1e3, thres2=5e1, weight=(1, 10),
                           method="naiveEuler", rescale_const=1, time_scale=10, buffer_start_steps=2):
    odefunc = RNNODE(input_dim=obs_dim, n_latent=n_latent, n_hidden=n_hidden)
    outputfunc = OutputNN(input_dim=obs_dim, n_latent=n_latent)
    params = list(odefunc.parameters()) + list(outputfunc.parameters())
    opt = optim.Adam(params, lr=lr)
    train_loss = []
    fit_err = []
    min_valid_loss = 1e5

    t1 = time.time()
    for itr in range(1, n_iter + 1):
        loss_epoch = 0.
        for batch_idx, samples in enumerate(train_dataloader):

            samples, train_ts_temp = samples[0], samples[1]
            h0 = torch.zeros(samples.shape[0], n_latent)
            loss = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
            opt.zero_grad()
            for t_ind in range(input_steps - 1):
                x_temp = samples[:, t_ind, :] * rescale_const
                deltat_mult = (train_ts_temp[:, t_ind + 1] - train_ts_temp[:, t_ind])
                if method == "naiveEuler":

                    h1 = h0 + odefunc(t=train_ts_temp[0, 0], x=x_temp, h=h0
                                      ) * deltat_mult.reshape(samples.shape[0], 1).repeat(1, n_latent) / time_scale
                # else:
                #     h1 = odeint(lambda t, h: odefunc(t=t, x=x_temp, h=h),
                #                 h0, train_ts_temp[0][t_ind:t_ind + 2] / time_scale, method=method)
                #     h1 = h1[-1, :]
                xhat = outputfunc(h1) / rescale_const
                loss = loss + torch.sum((xhat - samples[:, t_ind + 1, :]) ** 2
                                        * deltat_mult.reshape(samples.shape[0], 1).repeat(1, obs_dim)) / time_scale
                h0 = h1

            loss_epoch += loss.item()
            loss.backward()
            opt.step()

        _, _, valid_loss1, _ = pred_non_adap_models_grids(odefunc, outputfunc, valid_window, valid_ts, num_grid,
                                                          buffer_start_steps=buffer_start_steps, verbose=0,
                                                          rescale_const=rescale_const,
                                                          obs_dim=obs_dim, n_latent=n_latent, time_scale=time_scale)
        _, valid_loss2, _ = fit_non_adap_models_grids(odefunc, outputfunc, valid_window, valid_ts, num_grid,
                                                      buffer_start_steps=buffer_start_steps, verbose=0,
                                                      rescale_const=rescale_const,
                                                      obs_dim=obs_dim, n_latent=n_latent, time_scale=time_scale)
        valid_loss1, valid_loss2 = valid_loss1.mean(), valid_loss2.mean()
        valid_loss = weight[0] * valid_loss1 + weight[1] * valid_loss2
        fit_err.append(valid_loss)
        if valid_loss < min_valid_loss:
            torch.save([odefunc, outputfunc], 'saved_model2.pth')
            min_valid_loss = valid_loss

        t2 = time.time()
        remain_time = (t2 - t1) / itr * (n_iter - itr)
        train_loss.append(loss_epoch / num_train_windows)
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
    plt.plot(train_loss)
    plt.show()

    plt.plot(fit_err)
    plt.title('validation error')
    [odefunc, outputfunc] = torch.load('saved_model2.pth')
    _, _, valid_loss1, _ = pred_non_adap_models_grids(odefunc, outputfunc, valid_window, valid_ts, num_grid,
                                                      buffer_start_steps=buffer_start_steps, verbose=0,
                                                      rescale_const=rescale_const,
                                                      obs_dim=obs_dim, n_latent=n_latent, time_scale=time_scale)
    _, valid_loss2, _ = fit_non_adap_models_grids(odefunc, outputfunc, valid_window, valid_ts, num_grid,
                                                  buffer_start_steps=buffer_start_steps, verbose=0,
                                                  rescale_const=rescale_const,
                                                  obs_dim=obs_dim, n_latent=n_latent, time_scale=time_scale)
    valid_loss1, valid_loss2 = valid_loss1.mean(), valid_loss2.mean()
    valid_loss = weight[0] * valid_loss1 + weight[1] * valid_loss2
    plt.axhline(valid_loss, color='red', linestyle='--')
    plt.show()

    print('Training Neural ODE takes %.2f s' % (t3 - t1))

    return True, odefunc, outputfunc




