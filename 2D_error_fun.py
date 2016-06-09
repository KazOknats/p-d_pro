from string import letters
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import flopy
import flopy.utils.binaryfile as bf
from matplotlib.colors import BoundaryNorm
import modelz as zm

__author__ = 'stanko'

mod_nam = '2D_07'
mod_type = 'pod_r56'
w_dir = os.getcwd()
sim_dir = os.path.join('..', 'model', mod_nam)
res_dir = os.path.join(sim_dir, '_result')
mo_dir = os.path.join(res_dir, mod_type)
fig_dir = os.path.join(sim_dir, '_fig')

if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)

pdf_dir = os.path.join(fig_dir, '_pdf')
if not os.path.exists(pdf_dir):
    os.mkdir(pdf_dir)

mo_dir = os.path.join(res_dir, 'pod')
# import the snapshots of h
print('importing snapshots ... ')
fil_nam = 'SnapShots_h.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
ss_hp = np.genfromtxt(f, skip_header=1)

fil_nam = 'SnapShotsR.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
ss_hrp = np.genfromtxt(f, skip_header=1)
print(' ... done')


mo_dir = os.path.join(res_dir, 'pod-deim')
# import the snapshots of h
print('importing snapshots ... ')
fil_nam = 'SnapShots_h.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
ss_h = np.genfromtxt(f, skip_header=1)

fil_nam = 'SnapShotsR.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
ss_hr = np.genfromtxt(f, skip_header=1)
print(' ... done')

# import the snapshots of Ah
print('importing snapshots of Ah ... ')
fil_nam = 'SnapShots_Ah.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
ss_Ah = np.genfromtxt(f, skip_header=1)
print(' ... done')

print('importing snapshots of AR ... ')
fil_nam = 'SnapShotsAR.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
ss_Ahr = np.genfromtxt(f, skip_header=0)
print(' ... done')

print('importing basis of Ah ... ')
fil_nam = 'basis_Ah.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
basis_Ah = np.genfromtxt(f, skip_header=1)
didx = basis_Ah[0, ]
didx = [int(d) for d in didx]
print(' ... done')

nsp = 4
nts = 230
nss = 230
ny = 198
nx = 198
nz = 1

newparams = {'figure.dpi': 150, 'savefig.dpi': 150,
             'font.family': 'serif', 'pdf.compression': 0}
plt.rcParams.update(newparams)
X = np.arange(0, nx, 1)
Y = np.arange(0, ny, 1)
cmap = plt.get_cmap('Greys')

zhp = zm.Zrez(nts, nx, ny, ss_hp, ss_hrp)
plot_2d_nrmse(mod_nam, 'POD', 'h', X, Y, zhp.nrmse_xy, 6, 6, 12)
plt.close()
# zh = zm.Zmode(nx, ny, nz, nsp, nts, nss, )
zh = zm.Zrez(nts, nx, ny, ss_h, ss_hr)
plot_2d_nrmse(mod_nam, 'POD-DEIM', 'h', X, Y, zh.nrmse_xy, 6, 6, 12, didx)
plt.close()
#plt.clf()
zA = zm.Zrez(nts, nx, ny, ss_Ah, ss_Ahr)
plot_2d_nrmse(mod_nam, 'POD-DEIM', 'Ah', X, Y, zA.nrmse_xy, 6, 6, 12, didx)
plt.close()


def plot_2d_nrmse(nam, typ, ertp, x, y, z, sx, sy, sf, idx=None):
    nrm = mpl.colors.Normalize(vmin=z.min(), vmax=z.max())
    print('plot {} NRMSE for head over all time for each cell ... '.format(typ))
    zgrid = (z.reshape(nx, ny))
    figf = plt.figure(figsize=(sx, sy))
    axf = figf.add_subplot(1, 1, 1, aspect='equal')
    axf.set_title('{} Reduced Model Error, {} \n'.format(typ, ertp), fontsize=sf)
    imf = axf.pcolormesh(x, y, zgrid, cmap=cmap, norm=nrm)
    if idx:
        xy_loc = np.unravel_index(idx, (nx, ny))
        for k in np.arange(len(idx)):
            axf.plot(xy_loc[1][k], xy_loc[0][k], lw=1, marker='o', markersize=5,
                     markeredgewidth=0.5, markeredgecolor='black', markerfacecolor='red')
    axf.axis([x.min(), x.max(), y.min(), y.max()])
    axf.invert_yaxis()
    axf.set_xlabel('Column', fontsize=sf)
    axf.set_ylabel('Row', fontsize=sf)
    cbf = figf.colorbar(imf, ax=axf, shrink=0.9)
    cbf.set_label('NRMSE', fontsize=sf)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '{}_err_{}_{}.png'.format(nam, typ, ertp)), bbox_inches='tight')
    plt.savefig(os.path.join(pdf_dir, '{}_err_{}_{}.pdf'.format(nam, typ, ertp)), bbox_inches='tight')
    print(' ... done \n')

namfile = os.path.join(sim_dir, '{}.nam'.format(mod_nam))

# import model
print('loading model with flopy ... ')
mf = flopy.modflow.Modflow.load(namfile)
print(' ... done')

print('making model grid figures ... ')
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
modelmap = flopy.plot.ModelMap(sr=mf.sr)
# linecollection = modelmap.plot_grid(colors='black', alpha=0.5)
modelmap = flopy.plot.ModelMap(model=mf)
quadmesh = modelmap.plot_ibound()
quadmesh = modelmap.plot_bc('CHD')
quadmesh = modelmap.plot_bc('GHB')
quadmesh = modelmap.plot_bc('WEL')
quadmesh = modelmap.plot_bc('RIV')
quadmesh = modelmap.plot_bc('DRN')
# plt.show()
# save figure
fignam = mod_nam+'model_grid_05'
plt.savefig(os.path.join(fig_dir, fignam+'.png'))
plt.savefig(os.path.join(pdf_dir, fignam+'.pdf'))
print(' ... done')

DIV = float(nts) / 100.0
xx = np.arange(nts).reshape(nts, 1)
xx += 1
a = np.array(xx, dtype=np.float64)
xex = a / DIV

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
RMSE = -np.sort(-zhp.rmse_t)
plt.plot(xex, 100. * RMSE, 'b', lw=3, ls='-', alpha=0.4, label='POD')
RMSE2 = -np.sort(-zh.rmse_t)
plt.plot(xex, 100. * RMSE2, 'darkred', ls='--', lw=2, label='POD-DEIM(64)')
# ax.axis([0., 100,10, 3000])
# xticks( range(1,35,4) )
# RMSE3 = -np.sort(-zh3.rmse_t)
# plt.plot(xex, 100. * RMSE3, 'k', ls='-', lw=.75, label='POD-DEIM(30)')
plt.xlabel('Excedence [%]', fontsize=12)
plt.ylabel('RMSE [cm]', fontsize=12)
plt.legend(loc='upper right')
plt.savefig(os.path.join(fig_dir, 'h_exede_{}.png'.format(mod_nam)))
plt.close()


MAE = sum(zh.T, 0) / float(nts)
plt.clf()
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(111)
ax1.plot(np.arange(nts), zh2[:, 6715] * 100, 'k', lw=2, label='Head Error [row 67 column 82]')
ax2 = ax1.twinx()
ax2.plot(np.arange(nts), MAE, 'r', lw=1.5, label='MAE')
ax1.set_xlabel('Time [Days]', fontsize=12)
ax1.set_ylabel('Absolute error [cm]', fontsize=12)
ax2.set_ylabel('Mean Average Error', fontsize=12)
ax1.legend(loc='upper left')
ax2.legend()
plt.savefig(os.path.join(fig_dir, '2_errors.png'))

plt.close('all')
