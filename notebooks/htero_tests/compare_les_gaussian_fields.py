#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[2]
LES = ROOT / 'runs/source_heterogeneity_les_source_l200_cv2p0/dispersion/concentration.nc'
GAUSS = ROOT / 'runs/paired_gaussian_les_field_comparison/dispersion/concentration.nc'
OUT = Path(__file__).with_name('les_gaussian_field_comparison.pdf')

with Dataset(LES) as ds:
    x = np.asarray(ds['x'][:])
    y = np.asarray(ds['y'][:])
    lon = np.asarray(ds['longitude'][:])
    lat = np.asarray(ds['latitude'][:])
    les = np.asarray(ds['concentration'][:]).mean(axis=0)

with Dataset(GAUSS) as ds:
    glon = np.asarray(ds['longitude'][:])
    glat = np.asarray(ds['latitude'][:])
    gaussian_native = np.asarray(ds['concentration'][:]).mean(axis=0)

gaussian = griddata(
    np.c_[glon.ravel(), glat.ravel()],
    gaussian_native.ravel(),
    (lon, lat),
    method='linear',
    fill_value=0.0,
)
difference = gaussian - les
rmse = float(np.sqrt(np.mean(difference**2)))
corr = float(np.corrcoef(les.ravel(), gaussian.ravel())[0, 1])
bias = float(np.mean(difference))
mean_ratio = float(np.mean(gaussian) / np.mean(les))
peak_ratio = float(np.max(gaussian) / np.max(les))
amplitude_scale = float(np.sum(les * gaussian) / np.sum(gaussian**2))
scaled_rmse = float(np.sqrt(np.mean((amplitude_scale * gaussian - les) ** 2)))

vmax = np.percentile(np.r_[les.ravel(), gaussian.ravel()], 99.5)
dmax = np.percentile(np.abs(difference), 99.5)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
for axis, field, title in zip(
    ax[:2], [les, gaussian], ['LES nature field', 'Gaussian-plume field']
):
    image = axis.pcolormesh(
        x, y, field, shading='auto', cmap='viridis', vmin=0, vmax=vmax
    )
    axis.set(title=title, xlabel='LES downwind x (m)', ylabel='LES crosswind y (m)')
    fig.colorbar(image, ax=axis, label='CH4 enhancement (ng m$^{-3}$)')

image = ax[2].pcolormesh(
    x, y, difference, shading='auto', cmap='coolwarm', vmin=-dmax, vmax=dmax
)
ax[2].set(
    title='Gaussian minus LES',
    xlabel='LES downwind x (m)',
    ylabel='LES crosswind y (m)',
)
fig.colorbar(image, ax=ax[2], label='Difference (ng m$^{-3}$)')
fig.suptitle(
    f'Time-mean field comparison: RMSE={rmse:.0f} ng m$^{{-3}}$, '
    f'bias={bias:.0f}, correlation={corr:.2f}'
)
fig.savefig(OUT, bbox_inches='tight')
print(json.dumps({
    'rmse_ng_m3': rmse,
    'bias_ng_m3': bias,
    'correlation': corr,
    'mean_gaussian_to_les': mean_ratio,
    'peak_gaussian_to_les': peak_ratio,
    'best_fit_gaussian_scale': amplitude_scale,
    'amplitude_scaled_rmse_ng_m3': scaled_rmse,
    'les_mean_ng_m3': float(np.mean(les)),
    'gaussian_mean_ng_m3': float(np.mean(gaussian)),
    'les_peak_ng_m3': float(np.max(les)),
    'gaussian_peak_ng_m3': float(np.max(gaussian)),
    'output': str(OUT),
}))
