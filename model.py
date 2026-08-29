"""
Pure-numpy sea-ice concentration forecaster.

Replaces the PyTorch ConvLSTM with a physically-motivated persistence +
advection-diffusion model.  For a demo / no-GPU environment this gives
visually plausible results without needing torch.

The core idea: ice tends to persist (persistence baseline), slowly advance
in winter (linear trend per cell), and spread/diffuse at edges (Gaussian
blur).  That's enough to produce a believable 5-day forecast grid.
"""
import numpy as np


def _gaussian_blur(arr, sigma=1.0):
    """Simple 2D Gaussian blur using a convolution kernel."""
    k = int(np.ceil(sigma * 3)) * 2 + 1
    ax = np.arange(k) - k // 2
    kernel_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel_2d /= kernel_2d.sum()

    # manual 2D convolution (scipy-free)
    h, w = arr.shape
    kh, kw = kernel_2d.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(arr, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(arr)
    for i in range(kh):
        for j in range(kw):
            out += kernel_2d[i, j] * padded[i : i + h, j : j + w]
    return out


def forecast_sic(history, forecast_len=5, blur_sigma=1.2):
    """
    Pure-numpy SIC forecast.

    Parameters
    ----------
    history : ndarray, shape (n_days, H, W) – recent SIC values 0-100
    forecast_len : int – number of future days to predict
    blur_sigma : float – edge-diffusion strength per step

    Returns
    -------
    ndarray, shape (forecast_len, H, W) – predicted SIC 0-100
    """
    # Trend: linear extrapolation per cell from last few days
    recent = history[-min(10, len(history)):]
    t = np.arange(len(recent), dtype=np.float64)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()
    if t_var == 0:
        trend = np.zeros_like(recent[-1])
    else:
        slope = ((recent - recent.mean(axis=0, keepdims=True)) * (t - t_mean)[:, None, None]).sum(axis=0) / t_var
        trend = slope  # per-day change

    last = recent[-1].copy()
    preds = []
    for d in range(1, forecast_len + 1):
        base = last + trend * d
        # diffusion: ice spreads slightly each day
        diffused = _gaussian_blur(base, sigma=blur_sigma * d)
        # keep physical bounds
        preds.append(np.clip(diffused, 0, 100).astype(np.float32))

    return np.stack(preds)  # (forecast_len, H, W)


class SeaIceForecaster:
    """
    Drop-in replacement for the torch model class so existing call-sites
    that do ``SeaIceForecaster()`` still work.
    """

    def __init__(self, hidden_channels=16, forecast_len=5):
        self.forecast_len = forecast_len

    def predict(self, history):
        return forecast_sic(history, self.forecast_len)
