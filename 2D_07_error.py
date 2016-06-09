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
    cbf = figf.colorbar(imf, ax=axf, shrink=0.8)
    cbf.set_label('NRMSE', fontsize=sf)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '{}_err_{}_{}.png'.format(nam, typ, ertp)), bbox_inches='tight')
    plt.savefig(os.path.join(pdf_dir, '{}_err_{}_{}.pdf'.format(nam, typ, ertp)), bbox_inches='tight')
    print(' ... done \n')

mod_nam = '2D_07'
mod_type = 'pod_r56'
w_dir = os.getcwd()
sim_dir = os.path.join('..', 'model', mod_nam)
mo_dir = os.path.join(sim_dir, mod_type)
fig_dir = os.path.join(sim_dir, '_fig')

if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)

pdf_dir = os.path.join(fig_dir, '_pdf')
if not os.path.exists(pdf_dir):
    os.mkdir(pdf_dir)

# import the full model snapshots of h
print('importing snapshots of h ... ')
fil_nam = 'SnapShots_h.txt'
f = open(os.path.join(sim_dir, 'full', fil_nam), 'rb')
ss_h = np.genfromtxt(f, skip_header=1)
print(' ... done')

# import the pod-reduced model snapshots of h
print('importing POD snapshots of hr ... ')
mod_type = 'pod_r56'
mo_dir = os.path.join(sim_dir, mod_type)
fil_nam = 'SnapShotsR.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
ss_hrp = np.genfromtxt(f, skip_header=1)
print(' ... done')

# import the pod-deim-reduced model snapshots of h
print('importing DEIM snapshots of hr ... ')
mod_type = 'pod-deim_d100'
mo_dir = os.path.join(sim_dir, mod_type)
fil_nam = 'SnapShotsR.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
ss_hr = np.genfromtxt(f, skip_header=1)
print(' ... done')

# import the full model snapshots of A*h
print('importing snapshots of Ah ... ')
fil_nam = 'SnapShots_Ah.txt'
f = open(os.path.join(sim_dir, 'full', fil_nam), 'rb')
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

# import the full model snapshots of A*h for different deim model
mod_type = 'pod-deim_d150'
mo_dir = os.path.join(sim_dir, mod_type)
print('importing DEIM_2 snapshots of hr ... ')
fil_nam = 'SnapShotsR.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
ss_hr2 = np.genfromtxt(f, skip_header=1)
print(' ... done')

print('importing DEIM_2 snapshots of AR ... ')
fil_nam = 'SnapShotsAR.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
ss_Ahr2 = np.genfromtxt(f, skip_header=0)
print(' ... done')

print('importing DEIM_2 basis of Ah ... ')
fil_nam = 'basis_Ah.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
basis_Ah2 = np.genfromtxt(f, skip_header=1)
didx2 = basis_Ah2[0, ]
didx2 = [int(d) for d in didx2]
print(' ... done')

d = len(didx)
d2 = len(didx2)
nsp = 5
nts = 61
nss = 488
ny = 198
nx = 198
nz = 1

newparams = {'figure.dpi': 150, 'savefig.dpi': 300,
             'font.family': 'serif', 'font.size': 10, 'pdf.compression': 0}
plt.rcParams.update(newparams)
X = np.arange(0, nx, 1)
Y = np.arange(0, ny, 1)
cmap = plt.get_cmap('Greys')

# POD error in head
zh_p = zm.Zrez(nts, nx, ny, ss_h, ss_hrp)
plot_2d_nrmse(mod_nam, 'POD', 'h', X, Y, zh_p.nrmse_xy, 6, 6, 12)
plt.close()

# DEIM error in head
zh_d = zm.Zrez(nts, nx, ny, ss_h, ss_hr)
plot_2d_nrmse(mod_nam, 'POD-DEIM_d={}'.format(d), 'h', X, Y, zh_d.nrmse_xy, 6, 6, 12, didx)
plt.close()

# DEIM error in A*h
zA = zm.Zrez(nts, nx, ny, ss_Ah, ss_Ahr)
plot_2d_nrmse(mod_nam, 'POD-DEIM_d={}'.format(d), 'Ah', X, Y, zA.nrmse_xy, 6, 6, 12, didx)
plt.close('all')

# second DEIM model error in head
zh_d2 = zm.Zrez(nts, nx, ny, ss_h, ss_hr2)
plot_2d_nrmse(mod_nam, 'POD-DEIM_d={}'.format(d2), 'h', X, Y, zh_d2.nrmse_xy, 6, 6, 12, didx2)
plt.close()
# second DEIM model error in A*h
zA2 = zm.Zrez(nts, nx, ny, ss_Ah, ss_Ahr2)
plot_2d_nrmse(mod_nam, 'POD-DEIM_d={}'.format(d2), 'Ah', X, Y, zA2.nrmse_xy, 6, 6, 12, didx2)
plt.close('all')

namfile = os.path.join(sim_dir, '{}.nam'.format(mod_nam))

# import model
print('loading model with flopy ... ')
mf = flopy.modflow.Modflow.load(namfile)
print(' ... done')

print('making model grid figures ... ')
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
modelmap = flopy.plot.ModelMap(sr=mf.sr)
linecollection = modelmap.plot_grid(colors='black', lw=0.5, alpha=0.4)
modelmap = flopy.plot.ModelMap(model=mf)
quadmesh = modelmap.plot_bc('CHD')
quadmesh = modelmap.plot_bc('GHB')
quadmesh = modelmap.plot_bc('WEL')
quadmesh = modelmap.plot_bc('RIV')
quadmesh = modelmap.plot_bc('DRN')
# plt.show()
# save figure
fignam = mod_nam+'model_grid_07'
plt.savefig(os.path.join(fig_dir, fignam+'.png'))
plt.savefig(os.path.join(pdf_dir, fignam+'.pdf'))
print(' ... done')

DIV = float(nts) / 100.0
xx = np.arange(nts).reshape(nts, 1)
xx += 1
a = np.array(xx, dtype=np.float64)
xex = a / DIV

RMSE = -np.sort(-zh_p.rmse_t)
RMSE2 = -np.sort(-zh_d.rmse_t)
RMSE3 = -np.sort(-zh_d2.rmse_t)

td = r'POD-DEIM$_{d=100}$'
td2 = r'POD-DEIM$_{d=150}$'

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)
plt.plot(xex, 100. * RMSE, 'b', lw=3, ls='-', alpha=0.4, label='POD')
plt.plot(xex, 100. * RMSE2, 'darkred', ls='--', lw=2, label=td)
plt.plot(xex, 100. * RMSE3, 'k', ls='-', lw=.75, label=td2)
ax.set_yscale("log")
plt.xlabel('Excedence [%]', fontsize=12)
plt.ylabel('RMSE [cm]', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'h_exede_{}.png'.format(mod_nam)))
plt.savefig(os.path.join(pdf_dir, 'h_exede_{}.pdf'.format(mod_nam)))
plt.close()


MAE = np.sum(zh_d.err, axis=1) / float(zh_d.e_shape[1])
rowcol = zh_d.emx_loc[1]
row = zh_d.xy_loc[1]
col = zh_d.xy_loc[2]
lab = 'Head Error [row {} column {}]'.format(row, col)

plt.clf()

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(111)
ax1.plot(np.arange(nts), zh_d.err[:, rowcol], 'k', ls='--', lw=2, label=lab)
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim())
ax2.plot(np.arange(nts), MAE, 'r', lw=1.5, label='MAE')
ax1.set_xlabel('Time [Days]')
ax1.set_ylabel('Absolute error [cm]')
ax2.set_ylabel('Mean Absolute Error')
ax1.legend(loc=(0.05, 0.9))
ax2.legend(loc=(0.05, 0.8))
plt.savefig(os.path.join(fig_dir, '2_errors.png'))
plt.savefig(os.path.join(pdf_dir, '2_errors.pdf'))

# plot POD model errors
mod_type = 'POD'
MAE = np.sum(zh_p.err, axis=1) / float(zh_p.e_shape[1])
rowcol = zh_p.emx_loc[1]
row = zh_p.xy_loc[1]
col = zh_p.xy_loc[2]
lab = 'Head Error [row {} column {}]'.format(row, col)

fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(111)
ax1.plot(np.arange(nts), zh_p.err[:, rowcol], 'k', lw=1, label=lab)
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim())
ax2.plot(np.arange(nts), MAE, 'r', lw=1.5, alpha=0.6, label='MAE')
ax1.set_xlabel('Time step')
ax1.set_ylabel('Absolute error [m]')
ax2.set_ylabel('Mean Absolute Error [m]')
ax1.legend(loc=(0.05, 0.9))
ax2.legend(loc=(0.05, 0.8))
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, '2_errors_{}.png'.format(mod_type)), bbox_inches='tight')
plt.savefig(os.path.join(pdf_dir, '2_errors_{}.pdf'.format(mod_type)), bbox_inches='tight')

# plot DEIM model errors
mod_type = 'POD-DEIM_100'
MAE = np.sum(zh_d.err, axis=1) / float(zh_d.e_shape[1])
rowcol = zh_d.emx_loc[1]
row = zh_d.xy_loc[1]
col = zh_d.xy_loc[2]
lab = 'Head Error [row {} column {}]'.format(row, col)

fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(111)
ax1.plot(np.arange(nts), zh_d.err[:, rowcol], 'k', lw=1, label=lab)
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim())
ax2.plot(np.arange(nts), MAE, 'r', lw=1.5, alpha=0.6, label='MAE')
ax1.set_xlabel('Time step')
ax1.set_ylabel('Absolute error [m]')
ax2.set_ylabel('Mean Absolute Error [m]')
ax1.legend(loc=(0.05, 0.9))
ax2.legend(loc=(0.05, 0.8))
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, '2_errors_{}.png'.format(mod_type)), bbox_inches='tight')
plt.savefig(os.path.join(pdf_dir, '2_errors_{}.pdf'.format(mod_type)), bbox_inches='tight')

# plot second DEIM model errors
mod_type = 'POD-DEIM_150'
MAE = np.sum(zh_d2.err, axis=1) / float(zh_d2.e_shape[1])
rowcol = zh_d2.emx_loc[1]
row = zh_d2.xy_loc[1]
col = zh_d2.xy_loc[2]
lab = 'Head Error [row {} column {}]'.format(row, col)

fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(111)
ax1.plot(np.arange(nts), zh_d2.err[:, rowcol], 'k', lw=1, label=lab)
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim())
ax2.plot(np.arange(nts), MAE, 'r', lw=1.5, alpha=0.6, label='MAE')
ax1.set_xlabel('Time step')
ax1.set_ylabel('Absolute error [m]')
ax2.set_ylabel('Mean Absolute Error [m]')
ax1.legend(loc=(0.05, 0.9))
ax2.legend(loc=(0.05, 0.8))
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, '2_errors_{}.png'.format(mod_type)), bbox_inches='tight')
plt.savefig(os.path.join(pdf_dir, '2_errors_{}.pdf'.format(mod_type)), bbox_inches='tight')

namfile = os.path.join(sim_dir, '{}.nam'.format(mod_nam))
print('loading model with flopy ... ')
mf = flopy.modflow.Modflow.load(namfile)
print(' ... done')

plot_idx = 47
plot_time = 95
plot_idx2 = 52
plot_time2 = 96
plot_row = 63
plot_row2 = 78
bhd_file = os.path.join(sim_dir, mod_nam + '.bhd')
hdobj1 = bf.HeadFile(bhd_file, precision='single', verbose=True)
hdobj1.list_records()
hdobj1.get_kstpkper()
hdobj1.get_times()
rec1 = hdobj1.get_data(idx=plot_idx)
rec1_2 = hdobj1.get_data(idx=plot_idx2)

bhd_file = os.path.join(mo_dir, mod_nam + '.bhd')
hdobj2 = bf.HeadFile(bhd_file, precision='single', verbose=True)
rec2 = hdobj2.get_data(idx=plot_idx)
rec2_2 = hdobj2.get_data(idx=plot_idx2)

xy_axis = [0, 10000, -20, -4]
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title('Row {}'.format(plot_row), fontsize=12)
xs = flopy.plot.ModelCrossSection(model=mf, line={'row': plot_row-1})
linecollection = xs.plot_grid(alpha=0.2)
wt = xs.plot_surface(rec1, masked_values=[999.], color='blue', lw=4,
                     label='Full, {} days'.format(plot_time), alpha=0.3)
wt = xs.plot_surface(rec2, masked_values=[999.], color='black', lw=.75, ls='-',
                     label='POD-DEIM, {} days'.format(plot_time))
#                     marker='o', markersize=3, markerfacecolor='None')
ax1.axis(xy_axis)
# ax2.yaxis.tick_right()
# ax2.yaxis.set_label_position("right")
ax1.set_ylabel('Hydraulic Head (m)', fontsize=12)
ax1.legend(loc='lower right', fontsize=12)

xy_axis = [5000, 15000, -20, -4]
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title('Row {}'.format(plot_row2), fontsize=12)
xs = flopy.plot.ModelCrossSection(model=mf, line={'row': plot_row2-1})
linecollection = xs.plot_grid(alpha=0.2)
wt = xs.plot_surface(rec1, masked_values=[999.], color='blue', lw=4,
                     label='Full, {} days'.format(plot_time), alpha=0.3)
wt = xs.plot_surface(rec2, masked_values=[999.], color='black', lw=.75, ls='-',
                     label='POD-DEIM, {} days'.format(plot_time))
#                     marker='o', markersize=3, markerfacecolor='None')
ax2.axis(xy_axis)
# ax2.yaxis.tick_right()
# ax2.yaxis.set_label_position("right")
ax2.set_ylabel('Hydraulic Head (m)', fontsize=12)
ax2.set_xlabel('Horizontal Distance (m)', fontsize=12)
ax2.legend(loc='lower left', fontsize=12)
plt.savefig(os.path.join(fig_dir, 'wt_err_t{}_r{}.png'.format(plot_time, plot_row)), bbox_inches='tight')
plt.savefig(os.path.join(pdf_dir, 'wt_err_t{}_r{}.pdf'.format(plot_time, plot_row)), bbox_inches='tight')

xy_axis = [0, 10000, -20, -4]
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title('Row {}'.format(plot_row), fontsize=12)
xs = flopy.plot.ModelCrossSection(model=mf, line={'row': plot_row-1})
linecollection = xs.plot_grid(alpha=0.2)
wt = xs.plot_surface(rec1_2, masked_values=[999.], color='blue', lw=4,
                     label='Full, {} days'.format(plot_time2), alpha=0.3)
wt = xs.plot_surface(rec2_2, masked_values=[999.], color='black', lw=.75, ls='-',
                     label='POD-DEIM, {} days'.format(plot_time2))
#                     marker='o', markersize=3, markerfacecolor='None')
ax1.axis(xy_axis)
# ax2.yaxis.tick_right()
# ax2.yaxis.set_label_position("right")
ax1.set_ylabel('Hydraulic Head (m)', fontsize=12)
ax1.legend(loc='lower right', fontsize=12)

xy_axis = [5000, 15000, -20, -4]
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title('Row {}'.format(plot_row2), fontsize=12)
xs = flopy.plot.ModelCrossSection(model=mf, line={'row': plot_row2-1})
linecollection = xs.plot_grid(alpha=0.2)
wt = xs.plot_surface(rec1_2, masked_values=[999.], color='blue', lw=4,
                     label='Full, {} days'.format(plot_time2), alpha=0.3)
wt = xs.plot_surface(rec2_2, masked_values=[999.], color='black', lw=.75, ls='-',
                     label='POD-DEIM, {} days'.format(plot_time2))
#                     marker='o', markersize=3, markerfacecolor='None')
ax2.axis(xy_axis)
# ax2.yaxis.tick_right()
# ax2.yaxis.set_label_position("right")
ax2.set_ylabel('Hydraulic Head (m)', fontsize=12)
ax2.set_xlabel('Horizontal Distance (m)', fontsize=12)
ax2.legend(loc='lower left', fontsize=12)
plt.savefig(os.path.join(fig_dir, 'wt_err_t{}_r{}.png'.format(plot_time2, plot_row)), bbox_inches='tight')
plt.savefig(os.path.join(pdf_dir, 'wt_err_t{}_r{}.pdf'.format(plot_time2, plot_row)), bbox_inches='tight')


# plot four at a time
print('POD-DEIM reduced model error figures ... ')
idx_num = 47
day1 = np.int(hdobj1.get_times()[idx_num])
fig = plt.figure(figsize=(10, 10))
norm_1 = mpl.colors.Normalize(vmin=0.0, vmax=zh_d2.err[idx_num].max() * 100.0)
ax = fig.add_subplot(2, 2, 1, aspect='equal')
ax.set_title('Time: {} days \n'.format(day1), fontsize=10)
z_grid = (zh_d2.err[idx_num, :].reshape(nx, ny)) * 100.0
im = ax.pcolormesh(X, Y, z_grid, cmap=cmap, norm=norm_1)
ax.axis([X.min(), X.max(), Y.min(), Y.max()])
ax.invert_yaxis()
# ax.set_xlabel('Column', fontsize=10)
ax.set_ylabel('Row', fontsize=10)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
# cb.set_label('Absolute Error in head (cm)', fontsize=10)
plt.tight_layout()

norm_3 = mpl.colors.Normalize(vmin=0.0, vmax=zA2.err[idx_num].max())
ax3 = fig.add_subplot(2, 2, 3, aspect='equal')
# ax3.set_title('Time: {} days \n'.format(day1), fontsize=10)
z_grid = (zA2.err[idx_num, :].reshape(nx, ny))
im = ax3.pcolormesh(X, Y, z_grid, cmap=cmap, norm=norm_3)
ax3.axis([X.min(), X.max(), Y.min(), Y.max()])
ax3.invert_yaxis()
ax3.set_xlabel('Column', fontsize=10)
ax3.set_ylabel('Row', fontsize=10)
cb = fig.colorbar(im, ax=ax3, shrink=0.8)
# cb.set_label('Nonlinear Error', fontsize=10)
plt.tight_layout()

idx_num2 = 53
day2 = np.int(hdobj1.get_times()[idx_num2])
norm_2 = mpl.colors.Normalize(vmin=0.0, vmax=zh_d2.err[idx_num2].max() * 100.0)
ax2 = fig.add_subplot(2, 2, 2, aspect='equal')
ax2.set_title('Time: {} days \n'.format(day2), fontsize=10)
z_grid = (zh_d2.err[idx_num2, :].reshape(nx, ny)) * 100.0
im = ax2.pcolormesh(X, Y, z_grid, cmap=cmap, norm=norm_1)
ax2.axis([X.min(), X.max(), Y.min(), Y.max()])
ax2.invert_yaxis()
# ax2.set_xlabel('Column', fontsize=10)
# ax2.set_ylabel('Row', fontsize=10)
cb = fig.colorbar(im, ax=ax2, shrink=0.8)
cb.set_label('Absolute Error in head (cm)', fontsize=10)
plt.tight_layout()

norm_4 = mpl.colors.Normalize(vmin=0.0, vmax=zA2.err[idx_num2].max())
ax4 = fig.add_subplot(2, 2, 4, aspect='equal')
# ax4.set_title('Time: {} days \n'.format(day2), fontsize=10)
z_grid = (zA2.err[idx_num2, :].reshape(nx, ny))
im = ax4.pcolormesh(X, Y, z_grid, cmap=cmap, norm=norm_3)
ax4.axis([X.min(), X.max(), Y.min(), Y.max()])
ax4.invert_yaxis()
ax4.set_xlabel('Column', fontsize=10)
# ax4.set_ylabel('Row', fontsize=10)
cb = fig.colorbar(im, ax=ax4, shrink=0.8)
cb.set_label('Nonlinear Error', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, '4_err_d{}&{}.png'.format(day1, day2)), bbox_inches='tight')
plt.savefig(os.path.join(pdf_dir, '4_err_d{}&{}.pdf'.format(day1, day2)), bbox_inches='tight')
print(' ... done')

plt.close('all')
