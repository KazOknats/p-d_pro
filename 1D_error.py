
# import pandas as pd
# import seaborn as sns
# import matplotlib.cm as cm
# import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import os
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from __future__ import print_function


start = time.time()
def elapsed():
    return time.time() - start

# mod_nam = '2D_unconfined_03'
mod_nam = '1D_unconfined_02'
mod_type = 'revised_pod'
fil_dir = os.path.join('..', 'model', mod_nam)
res_dir = os.path.join(fil_dir, '_result')
mo_dir = os.path.join(res_dir, mod_type)

# import the snapshots of h
fil_nam = 'SnapShots_h.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
ss_h = np.genfromtxt(f, skip_header=1)

fil_nam = 'SnapShotsR.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
ss_hr = np.genfromtxt(f, skip_header=1)

zh = np.abs(ss_h - ss_hr)
# zhlog10 = np.log10(zh)
hr_min = ss_hr.min()
h_min = ss_h.min()
zh_min, zh_max = np.abs(zh).min(), np.abs(zh).max()
zhmx_loc = np.unravel_index(np.argmax(zh), zh.shape)
zh_n = zh.size
zh_rmse = np.linalg.norm(zh) / np.sqrt(zh_n)
zh_nrmse = zh_rmse/zh_max

print ('*** error in h ***')
print (' min head (full): ', h_min)
print (' min head (red): ', hr_min)
print (' size: ', zh_n)
print (' min, max: ', zh_min, zh_max)
print (' max loc: ', zhmx_loc)
print (' RMSE: ', zh_rmse)
print (' NRMSE: ', zh_nrmse)

nts = 90
t = 15
ny = 1
nx = 200
n = 20

X = np.arange(0, nx, 1)
Y = np.arange(0, nts, 1)
dx, dt = 1, 1
y, x = np.mgrid[slice(1, nx + dx, dx),
                slice(1, nts + dt, dt)]

levels = MaxNLocator(nbins=30).tick_values(zh_min, zh_max)
cmap = plt.get_cmap('bwr')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

norm = mpl.colors.Normalize(vmin=zh_min * 100.0, vmax=zh_max * 100.0)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
plt.title('POD Reduced Model Error \n', fontsize=14)
im = ax.pcolormesh(X, Y, zh[:, :] * 100.0, cmap=cmap, norm=norm)
plt.xlim(X.min(), X.max())
plt.ylim(Y.min(), Y.max())
plt.xlabel('Horizontal Distance (m)', fontsize=12)
plt.ylabel('Time (days)', fontsize=12)
cb = fig.colorbar(im, ax=ax, shrink=0.9)
cb.set_label('Absolute Error in head (cm)', fontsize=12)
plt.savefig(os.path.join(fil_dir, 'fig', '0_err_{0}.png'.format(mod_type)))
plt.close()

# repeat for pod-deim

mod_nam = '1D_unconfined_02'
mod_type = 'revised_pod-deim'
mo_dir = os.path.join(res_dir, mod_type)

# snapshots of h are the same as before
# reduced snapshots
fil_nam = 'SnapShotsR.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
ss_hr = np.genfromtxt(f, skip_header=1)

zh = np.abs(ss_h - ss_hr)
# zhlog10 = np.log10(zh)
hr_min = ss_hr.min()
h_min = ss_h.min()
zh_min, zh_max = np.abs(zh).min(), np.abs(zh).max()
zhmx_loc = np.unravel_index(np.argmax(zh), zh.shape)
zh_n = zh.size
zh_rmse = np.linalg.norm(zh) / np.sqrt(zh_n)
zh_nrmse = zh_rmse/zh_max

print ('*** error in h ***')
print (' min head (full): ', h_min)
print (' min head (red): ', hr_min)
print (' size: ', zh_n)
print (' min, max: ', zh_min, zh_max)
print (' max loc: ', zhmx_loc)
print (' RMSE: ', zh_rmse)
print (' NRMSE: ', zh_nrmse)

nts = 90
t = 15
ny = 1
nx = 200
n = 20

X = np.arange(0, nx, 1)
Y = np.arange(0, nts, 1)
dx, dt = 1, 1
y, x = np.mgrid[slice(1, nx + dx, dx),
                slice(1, nts + dt, dt)]

levels = MaxNLocator(nbins=30).tick_values(zh_min, zh_max)
cmap = plt.get_cmap('bwr')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

norm = mpl.colors.Normalize(vmin=zh_min * 100.0, vmax=zh_max * 100.0)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
plt.title('POD-DEIM Reduced Model Error \n', fontsize=14)
im = ax.pcolormesh(X, Y, zh[:, :] * 100.0, cmap=cmap, norm=norm)
plt.xlim(X.min(), X.max())
plt.ylim(Y.min(), Y.max())
plt.xlabel('Horizontal Distance (m)', fontsize=12)
plt.ylabel('Time (days)', fontsize=12)
cb = fig.colorbar(im, ax=ax, shrink=0.9)
cb.set_label('Absolute Error in head (cm)', fontsize=12)
plt.savefig(os.path.join(fil_dir, 'fig', '0_err_{0}.png'.format(mod_type)))
plt.close()

# import the snapshots of Ah
fil_nam = 'SnapShots_Ah.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
ss_Ah = np.genfromtxt(f, skip_header=1)

fil_nam = 'SnapShotsAR.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
ss_Ahr = np.genfromtxt(f)

zA = np.abs(ss_Ah - ss_Ahr)
# zAlog10 = np.log10(zA)
zA_min, zA_max = np.abs(zA).min(), np.abs(zA).max()
zAmx_loc = np.unravel_index(np.argmax(zA), zA.shape)

zA_n = zA.size
zA_rmse = np.linalg.norm(zA) / np.sqrt(zA_n)
zA_nrmse = zA_rmse/zA_max

print ('*** error in A*h ***')
print (' size: ', zA_n)
print (' min, max: ', zA_min, zA_max)
print (' max loc: ', zAmx_loc)
print (' RMSE: ', zA_rmse)
print (' NRMSE: ', zA_nrmse)

norm = mpl.colors.Normalize(vmin=zA_min, vmax=zA_max)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
plt.title('POD-DEIM Reduced Model Error \n', fontsize=14)
im = ax.pcolormesh(X, Y, zA[:, :], cmap=cmap, norm=norm)
plt.xlim(X.min(), X.max())
plt.ylim(Y.min(), Y.max())
plt.xlabel('Horizontal Distance (m)', fontsize=12)
plt.ylabel('Time (days)', fontsize=12)
cb = fig.colorbar(im, ax=ax, shrink=0.9)
cb.set_label('Nonlinear Error', fontsize=12)
plt.savefig(os.path.join(fil_dir, 'fig', 'Ah_err_{0}.png'.format(mod_type)))
plt.close()
