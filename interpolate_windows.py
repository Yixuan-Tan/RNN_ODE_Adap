import torch
from scipy.interpolate import interp1d

def interpolate_window(window, ts, ts0, interp_kind, obs_dim=2):
    window0 = torch.zeros(window.shape[0], ts0.shape[1], window.shape[2])
    for i in range(window.shape[0]):
        xk = ts[i, :]
        x2 = ts0[i, :]
        for k in range(obs_dim):
            yk = window[i, :, k]
            f = interp1d(xk, yk, kind=interp_kind)
            window0[i, :, k] = torch.tensor(f(x2))
    return window0


def interpolate_window_adap(window, ts, window_len, ts0, interp_kind, obs_dim=2):
    window0 = torch.zeros(window.shape[0], ts0.shape[1], window.shape[2])
    for i in range(window.shape[0]):
        xk = ts[i, :window_len[i]]
        x2 = ts0[i, :]
        for k in range(obs_dim):
            yk = window[i, :window_len[i], k]
            f = interp1d(xk, yk, kind=interp_kind)
            window0[i, :, k] = torch.tensor(f(x2))
    return window0
