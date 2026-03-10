import numpy as np
import pandas as pd


def curve_auc(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    area = np.trapz(y, x)
    width = x.max() - x.min()
    return area / width if width > 0 else np.nan


def noise_at_threshold(x, y, threshold):
    x = np.asarray(x)
    y = np.asarray(y)
    idx = np.where(y < threshold)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return float(x[0])
    x0, x1 = x[i - 1], x[i]
    y0, y1 = y[i - 1], y[i]
    if y1 == y0:
        return float(x1)
    t = (threshold - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0))


def summarize_curves(curves_by_model, noise_grid):
    rows = []
    for model_name, curve in curves_by_model.items():
        rows.append(
            {
                "model": model_name,
                "AUC": curve_auc(noise_grid, curve),
                "noise@90%": noise_at_threshold(noise_grid, curve, 0.90),
                "noise@50%": noise_at_threshold(noise_grid, curve, 0.50),
                "noise@20%": noise_at_threshold(noise_grid, curve, 0.20),
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("AUC", ascending=False)


def bootstrap_ci_mean(samples, n_boot=2000, alpha=0.10, seed=123):
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples)
    n = len(samples)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = samples[idx].mean()
    lo = np.quantile(boot, alpha / 2)
    hi = np.quantile(boot, 1 - alpha / 2)
    return lo, hi


def expected_calibration_error(conf, correct, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(conf, bins[1:-1], right=True)
    ece = 0.0
    acc_bin = []
    conf_bin = []
    cnt_bin = []
    for b in range(n_bins):
        m = bin_ids == b
        if m.sum() == 0:
            acc_bin.append(np.nan)
            conf_bin.append(np.nan)
            cnt_bin.append(0)
            continue
        acc_b = correct[m].mean()
        conf_b = conf[m].mean()
        w = m.mean()
        ece += w * abs(acc_b - conf_b)
        acc_bin.append(acc_b)
        conf_bin.append(conf_b)
        cnt_bin.append(m.sum())
    return ece, np.array(acc_bin), np.array(conf_bin), np.array(cnt_bin), bins
