import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import os
import flopy
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import flopy.utils.formattedfile as ff
from __future__ import print_function

start = time.time()
def elapsed():
    return time.time() - start

# mod_nam = '2D_unconfined_03'
mod_nam = '1D_02'
mod_type = 'pod'
w_dir = os.getcwd()
fil_dir = os.path.join('..', 'model', mod_nam)
res_dir = os.path.join(fil_dir, '_result')
mo_dir = os.path.join(res_dir, mod_type)
sim_dir = os.path.join('..', 'model', mod_nam)
namfile = os.path.join(sim_dir, '{}.nam'.format(mod_nam))
fig_dir = os.path.join(sim_dir, 'fig')
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)

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

# import model
print('loading model with flopy ... ')
mf = flopy.modflow.Modflow.load(namfile)
print(' ... done')

font = {'family': 'serif',
        'color':  'navy',
        'weight': 'semibold',
        'size': 14,
        'backgroundcolor': 'white',
        }

fhd_file = os.path.join(res_dir, 'Full', mod_nam + '.fhd')
hdobj1 = ff.FormattedHeadFile(fhd_file, precision='single', verbose=True)
rec1 = hdobj1.get_data(idx=60)
rec1_2 = hdobj1.get_data(idx=32)

fhd_file = os.path.join(res_dir, mod_type, mod_nam + '.fhd')
hdobj2 = ff.FormattedHeadFile(fhd_file, precision='single', verbose=True)
rec2 = hdobj2.get_data(idx=60)
rec2_2 = hdobj2.get_data(idx=32)

levels = np.arange(-50, 0, 2)
norm = mpl.colors.Normalize(vmin=zh_min * 100.0, vmax=zh_max * 100.0)
# plt.rcdefaults()
newparams = {'figure.dpi': 150, 'savefig.dpi': 300,
             'font.family': 'serif', 'pdf.compression': 0}
plt.rcParams.update(newparams)

fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('{} Reduced Model Error \n'.format(mod_type), fontsize=14)
im = ax1.pcolormesh(X, Y, zh[:, :] * 100.0, cmap='Greys', norm=norm)
ax1.set_xlim(X.min(), X.max())
ax1.set_ylim(Y.min(), Y.max())
ax1.set_xlabel('Horizontal Distance (m)', fontsize=12)
ax1.set_ylabel('Time (days)', fontsize=12)
cb = fig.colorbar(im, ax=ax1, shrink=0.9)
cb.set_label('Absolute Error in head (cm)', fontsize=12)

ax2 = fig.add_subplot(2, 2, 2)
# ax2.set_title('Time = 61 Days', fontsize=14)
xs = flopy.plot.ModelCrossSection(model=mf, line={'row': 0})
wt = xs.plot_surface(rec1, masked_values=[999.], color='blue', lw=1, label='Full, 61 days', alpha=0.5)
wt = xs.plot_surface(rec2, masked_values=[999.], color='black', lw=2, ls='--', label='Reduced, 61 days')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_ylabel('Hydraulic Head (m)', fontsize=12)
ax2.legend(loc='lower left', fontsize=12)

ax3 = fig.add_subplot(2, 2, 4)
# ax3.set_title('Time = 33 Days', fontsize=14)
xs = flopy.plot.ModelCrossSection(model=mf, line={'row': 0})
wt = xs.plot_surface(rec1_2, masked_values=[999.], color='blue', lw=1, label='Full, 33 days', alpha=0.5)
wt = xs.plot_surface(rec2_2, masked_values=[999.], color='black', lw=2, ls='--', label='Reduced, 33 days')
ax3.set_xlabel('Horizontal Distance (m)', fontsize=12)
ax3.yaxis.tick_right()
ax3.yaxis.set_label_position("right")
ax3.set_ylabel('Hydraulic Head (m)', fontsize=12)
ax3.legend(loc='lower left', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(fil_dir, 'fig', '3_err_{0}.png'.format(mod_type)))

plt.clf()
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
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
plt.title('POD-DEIM Reduced Model Error \n', fontsize=14)
im = ax.pcolormesh(X, Y, zA[:, :], cmap='Greys', norm=norm)
plt.xlim(X.min(), X.max())
plt.ylim(Y.min(), Y.max())
plt.xlabel('Horizontal Distance (m)', fontsize=12)
plt.ylabel('Time (days)', fontsize=12)
cb = fig.colorbar(im, ax=ax, shrink=0.9)
cb.set_label('Nonlinear Error', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(fil_dir, 'fig', '4_Ah_err_{0}.png'.format(mod_type)))

plt.close('all')
