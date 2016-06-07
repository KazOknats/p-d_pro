from string import letters
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import flopy
import flopy.utils.binaryfile as bf
from matplotlib.colors import BoundaryNorm


__author__ = 'stanko'

mod_nam = '2D_06'
mod_type = 'pod-deim'
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

# import the snapshots of h
print('importing snapshots ... ')
fil_nam = 'SnapShots_h.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
ss_h = np.genfromtxt(f, skip_header=1)

fil_nam = 'SnapShotsR.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
ss_hr = np.genfromtxt(f, skip_header=1)
print(' ... done')

nts = 230
nss = 230
ny = 198
nx = 198

zh = np.abs(ss_h - ss_hr)
# zhlog10 = np.log10(zh)
hr_min = ss_hr.min()
h_min = ss_h.min()
h_max = ss_h.max()
zh_min, zh_max = np.abs(zh).min(), np.abs(zh).max()
zhmx_loc = np.unravel_index(np.argmax(zh), zh.shape)
xy_loc = np.unravel_index(np.argmax(zh), (nts, nx, ny))
zh_n = zh.size
zh_rmse = np.linalg.norm(zh) / np.sqrt(zh_n)
zh_rmse_xy = np.linalg.norm(zh,axis=0) / np.sqrt(zh_n)
zh_rmse_t = np.linalg.norm(zh,axis=1) / np.sqrt(zh_n)
zh_nrmse = zh_rmse/zh_max
zh_nrmse_xy = zh_rmse_xy/zh_max
zh_nrmse_t = zh_rmse_t/zh_max

print ('*** {}, {} error in h ***'.format(mod_nam, mod_type))
print (' min head (full): ', h_min)
print (' min head (red): ', hr_min)
print (' size: ', zh_n)
print (' shape: ', zh.shape)
print (' min, max: ', zh_min, zh_max)
print (' max loc: ', zhmx_loc)
print (' max loc (t,x,y): ', xy_loc)
print ('AVG RMSE: ', zh_rmse.mean())
print (' NRMSE: ', zh_nrmse)

mod_type = 'pod-deim'
mo_dir = os.path.join(res_dir, mod_type)
fil_nam = 'SnapShotsR.txt'
print('importing snapshots of hR... ')
f = open(os.path.join(mo_dir, fil_nam), 'rb')
ss_hr2 = np.genfromtxt(f, skip_header=1)
print(' ... done')

zh2 = np.abs(ss_h - ss_hr2)
hr_min = ss_hr2.min()
zh2_min, zh2_max = np.abs(zh2).min(), np.abs(zh2).max()
zh2mx_loc = np.unravel_index(np.argmax(zh2), zh2.shape)
xy_loc = np.unravel_index(np.argmax(zh2), (nts, nx, ny))
zh2_n = zh2.size
zh2_rmse = np.linalg.norm(zh2) / np.sqrt(zh2_n)
zh2_rmse_xy = np.linalg.norm(zh2, axis=0) / np.sqrt(zh2_n)
zh2_rmse_t = np.linalg.norm(zh2, axis=1) / np.sqrt(zh2_n)
zh2_nrmse = zh2_rmse/zh2_max
zh2_nrmse_xy = zh2_rmse_xy/zh2_max
zh2_nrmse_t = zh2_rmse_t/zh2_max

print ('*** {}, {} error in h ***'.format(mod_nam, mod_type))
print (' min head (full): ', h_min)
print (' min head (red): ', hr_min)
print (' size: ', zh2_n)
print (' shape: ', zh2.shape)
print (' min, max: ', zh2_min, zh2_max)
print (' max loc: ', zh2mx_loc)
print (' max loc (t,x,y): ', xy_loc)
print ('AVG RMSE: ', zh2_rmse.mean())
print (' NRMSE: ', zh2_nrmse)

mod_type = 'pod-deim_30'
mo_dir = os.path.join(res_dir, mod_type)
fil_nam = 'SnapShotsR.txt'
f = open(os.path.join(mo_dir, fil_nam), 'rb')
ss_hr3 = np.genfromtxt(f, skip_header=1)

zh3 = np.abs(ss_h - ss_hr3)
hr_min = ss_hr3.min()
zh3_min, zh3_max = np.abs(zh3).min(), np.abs(zh3).max()
zh3mx_loc = np.unravel_index(np.argmax(zh3), zh3.shape)
xy_loc = np.unravel_index(np.argmax(zh3), (nts, nx, ny))
zh3_n = zh3.size
zh3_rmse = np.linalg.norm(zh3) / np.sqrt(zh3_n)
zh3_rmse_xy = np.linalg.norm(zh3,axis=0) / np.sqrt(zh3_n)
zh3_rmse_t = np.linalg.norm(zh3,axis=1) / np.sqrt(zh3_n)
zh3_nrmse = zh3_rmse/zh3_max
zh3_nrmse_xy = zh3_rmse_xy/zh3_max
zh3_nrmse_t = zh3_rmse_t/zh3_max

print ('*** {}, {} error in h ***'.format(mod_nam, mod_type))
print (' min head (full): ', h_min)
print (' min head (red): ', hr_min)
print (' size: ', zh3_n)
print (' shape: ', zh3.shape)
print (' min, max: ', zh3_min, zh3_max)
print (' max loc: ', zh3mx_loc)
print (' max loc (t,x,y): ', xy_loc)
print ('AVG RMSE: ', zh3_rmse.mean())
print (' NRMSE: ', zh3_nrmse)

plt.hist(zh_rmse_t)
plt.hist(zh2_rmse_t)

# import the snapshots of A* h
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
didx = [int(z) for z in didx]
print(' ... done')

zA = np.abs(ss_Ah - ss_Ahr)
# zAlog10 = np.log10(zA)
Ahr_min = ss_Ahr.min()
Ah_min = ss_Ah.min()
zA_min, zA_max = np.abs(zA).min(), np.abs(zA).max()
zAmx_loc = np.unravel_index(np.argmax(zA), zA.shape)
xy_loc = np.unravel_index(np.argmax(zA), (nts, nx, ny))
zA_n = zA.size
zA_rmse = np.linalg.norm(zA) / np.sqrt(zA_n)
zA_rmse_xy = np.linalg.norm(zA,axis=0) / np.sqrt(zA_n)
zA_rmse_t = np.linalg.norm(zA,axis=1) / np.sqrt(zA_n)
zA_nrmse = zA_rmse/zA_max
zA_nrmse_xy = zA_rmse_xy/zA_max
zA_nrmse_t = zA_rmse_t/zA_max

print ('*** error in A*h ***')
print (' min A*h (full): ', Ah_min)
print (' min A*h (red): ', Ahr_min)
print (' size: ', zA_n)
print (' shape: ', zA.shape)
print (' min, max: ', zA_min, zA_max)
print (' max loc: ', zAmx_loc)
print (' max loc (t,x,y): ', xy_loc)
print (' RMSE: ', zA_rmse)
print (' NRMSE: ', zA_nrmse)


# PLOTTING
# plt.rcdefaults()
newparams = {'figure.dpi': 150, 'savefig.dpi': 300,
             'font.family': 'serif', 'pdf.compression': 0}
plt.rcParams.update(newparams)
X = np.arange(0, nx, 1)
Y = np.arange(0, ny, 1)
cmap = plt.get_cmap('Greys')

# Plot POD Model first

norm = mpl.colors.Normalize(vmin=zh_nrmse_xy.min(), vmax=zh_nrmse_xy.max())
print('plot {} NRMSE for head over all time for each cell ... '.format(mod_type))
z_grid = (zh_nrmse_xy.reshape(nx, ny))
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
plt.title('POD Reduced Model Error \n', fontsize=14)
im = ax.pcolormesh(X, Y, z_grid, cmap=cmap, norm=norm)
ax.axis([X.min(), X.max(), Y.min(), Y.max()])
ax.invert_yaxis()
plt.xlabel('Column', fontsize=12)
plt.ylabel('Row', fontsize=12)
cb = fig.colorbar(im, ax=ax, shrink=0.9)
cb.set_label('NRMSE', fontsize=12)
plt.savefig(os.path.join(fig_dir, '2d_err_{}.png'.format(mod_type)), bbox_inches='tight')
plt.savefig(os.path.join(pdf_dir, '2d_err_{}.pdf'.format(mod_type)), bbox_inches='tight')
print(' ... done \n')
plt.close()

# Plot POD-DEIM
xy_loc = np.unravel_index(didx, (nx, ny))
print('plot POD-DEIM NRMSE for head over all time for each cell ... ')
z_grid = (zh2_nrmse_xy.reshape(nx, ny))
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
plt.title('POD-DEIM Reduced Model Error \n', fontsize=14)
im = ax.pcolormesh(X, Y, z_grid, cmap=cmap, norm=norm)
for z in np.arange(len(didx)):
    plt.plot(xy_loc[1][z], xy_loc[0][z], lw=1, marker='o', markersize=5,
             markeredgewidth=0.5, markeredgecolor='black', markerfacecolor='red')
ax.axis([X.min(), X.max(), Y.min(), Y.max()])
ax.invert_yaxis()
plt.xlabel('Column', fontsize=12)
plt.ylabel('Row', fontsize=12)
cb = fig.colorbar(im, ax=ax, shrink=0.9)
cb.set_label('NRMSE', fontsize=12)
plt.savefig(os.path.join(fig_dir, 'POD-DEIM_head_err_nrmse.png'), bbox_inches='tight')
plt.savefig(os.path.join(pdf_dir, 'POD-DEIM_head_err_nrmse.pdf'), bbox_inches='tight')
print(' ... done \n')
plt.close()

print('plot NRMSE for A*h over all time for each cell ... ')
norm = mpl.colors.Normalize(vmin=zA_nrmse_xy.min(), vmax=zA_nrmse_xy.max())
z_grid = (zA_nrmse_xy.reshape(nx, ny))
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
plt.title('POD-DEIM Reduced Model Error, A*h \n', fontsize=14)
im = ax.pcolormesh(X, Y, z_grid, cmap=cmap, norm=norm)
for z in np.arange(len(didx)):
    plt.plot(xy_loc[1][z], xy_loc[0][z], lw=1, marker='o', markersize=5,
             markeredgewidth=0.5, markeredgecolor='black', markerfacecolor='red')
ax.axis([X.min(), X.max(), Y.min(), Y.max()])
ax.invert_yaxis()
plt.xlabel('Column', fontsize=12)
plt.ylabel('Row', fontsize=12)
cb = fig.colorbar(im, ax=ax, shrink=0.9)
cb.set_label('NRMSE', fontsize=12)
plt.savefig(os.path.join(fig_dir, 'Ah_err_nrmse.png'), bbox_inches='tight')
plt.savefig(os.path.join(pdf_dir, 'Ah_err_nrmse.pdf'), bbox_inches='tight')
print(' ... done \n')
plt.close()

[2016-05-30]
namfile = os.path.join(sim_dir, '{}.nam'.format(mod_nam))
print('loading model with flopy ... ')
mf = flopy.modflow.Modflow.load(namfile)
print(' ... done')

plot_time = 65.
plot_time2 = 95.

bhd_file = os.path.join(sim_dir, mod_nam + '.bhd')
hdobj1 = bf.HeadFile(bhd_file, precision='single', verbose=True)
hdobj1.list_records()
rec1 = hdobj1.get_data(totim=plot_time)
rec1_2 = hdobj1.get_data(totim=plot_time2)

bhd_file = os.path.join(mo_dir, mod_nam + '.bhd')
hdobj2 = bf.HeadFile(bhd_file, precision='single', verbose=True)
rec2 = hdobj2.get_data(totim=plot_time)
rec2_2 = hdobj2.get_data(totim=plot_time2)


plot_row = 63
plot_row2 = 78
xy_axis = [0, 10000, -50, -4]
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

xy_axis = [5000, 15000, -50, -4]
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
ax2.legend(loc='lower right', fontsize=12)
plt.savefig(os.path.join(fig_dir, 'wt_err_t{}_r{}.png'.format(plot_time, plot_row)), bbox_inches='tight')
plt.savefig(os.path.join(pdf_dir, 'wt_err_t{}_r{}.pdf'.format(plot_time, plot_row)), bbox_inches='tight')


ax3 = fig.add_subplot(2, 2, 4)
# ax3.set_title('Time = 33 Days', fontsize=14)
xs = flopy.plot.ModelCrossSection(model=mf, line={'row': 0})
wt = xs.plot_surface(rec1_2, masked_values=[999.], color='blue', lw=1, label='Full, 33 days', alpha=0.5)
wt = xs.plot_surface(rec2_2, masked_values=[999.], color='black', lw=2, ls='--', label='POD, 33 days')
ax3.set_xlabel('Horizontal Distance (m)', fontsize=12)
ax3.yaxis.tick_right()
ax3.yaxis.set_label_position("right")
ax3.set_ylabel('Hydraulic Head (m)', fontsize=12)
ax3.legend(loc='lower left', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, '3_err_{0}.png'.format(mod_type)), bbox_inches='tight')
plt.savefig(os.path.join(pdf_dir, '3_err_{0}.pdf'.format(mod_type)), bbox_inches='tight')

# ____________________
rcdef = mpl.rcParams.copy()
# print(rcdef)
dpi = 300
newparams = {'figure.dpi':150, 'savefig.dpi':300, 'font.size':10}
# Update the global rcParams dictionary with the new parameter choices
# Before doing this, we reset rcParams to its default again, just in case
plt.rcParams.update(rcdef)
plt.rcParams.update(newparams)

row_num = 24

# didx = [1151, 1150, 1296, 1104, 1495, 1103, 1550, 1302]
# xy_loc = np.unravel_index(didx, (nx, ny))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
plt.title('POD-DEIM Reduced Model Error, Time: {} days \n'.format(row_num), fontsize=14)
z_grid = (zh[row_num, :].reshape(nx, ny)) * 100.0
im = ax.pcolormesh(X, Y, z_grid, cmap=cmap, norm=norm)
#for z in np.arange(len(didx)):
#    plt.plot(xy_loc[1][z], 99 - xy_loc[0][z], lw=1, marker='x', markersize=5,
#             markeredgewidth=0.5, markeredgecolor='black', markerfacecolor='none')
ax.axis([X.min(), X.max(), Y.min(), Y.max()])
ax.invert_yaxis()
plt.xlabel('Column', fontsize=12)
plt.ylabel('Row', fontsize=12)
cb = fig.colorbar(im, ax=ax, shrink=0.9)
cb.set_label('Absolute Error in head (cm)', fontsize=12)
plt.savefig(os.path.join(fig_dir, 'test_err_t{0}.png'.format(row_num)), bbox_inches='tight')
plt.savefig(os.path.join(pdf_dir, 'test_err_t{0}.pdf'.format(row_num)), bbox_inches='tight')

plt.close()

norm = mpl.colors.Normalize(vmin=zh_min * 100.0, vmax=zh_max * 100.0)

DIV = float(nts) / 100.0
x = np.arange(nts).reshape(nts, 1)
x += 1
a = np.array(x, dtype=np.float64)
xex = a / DIV

zh_rmse_t

MAE = sum(zh.T, 0) / float(nts)
RMSE = np.sqrt(sum(zh.T, 0) / float(nts))
RMSE2 = np.sqrt(sum(zh2.T, 0) / float(nts))
RMSE3 = np.sqrt(sum(zh3.T, 0) / float(nts))
NRMSE = 100 * RMSE / (h_max - h_min)

DEIM_18 = 'POD-DEIM(18)'
DEIM_20 = 'POD-DEIM(20)'
plt.style.use('ggplot')

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
RMSE = -np.sort(-zh_rmse_t)
plt.plot(xex, 100. * RMSE, 'b', lw=3, ls='-', alpha=0.4, label='POD')
RMSE2 = -np.sort(-zh2_rmse_t)
plt.plot(xex, 100. * RMSE2, 'darkred', ls='--', lw=2, label='POD-DEIM(40)')
#ax.axis([0., 100,10, 3000])
#xticks( range(1,35,4) )
RMSE3 = -np.sort(-zh3_rmse_t)
plt.plot(xex, 100. * RMSE3, 'k', ls='-', lw=.75, label='POD-DEIM(30)')
plt.xlabel('Excedence [%]', fontsize=12)
plt.ylabel('RMSE [cm]', fontsize=12)
plt.legend(loc='upper right')
plt.savefig(os.path.join(fig_dir, 'h_exede_{}.png'.format(mod_nam)))
plt.close()

# --------------------------------------------------------------------------------

MAE = sum(zA.T, 0) / float(nts)
RMSE = np.sqrt(sum(zA.T, 0) / float(nts))
NRMSE = 100 * RMSE / (zA_max - zA_min)

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
RMSE = -np.sort(-RMSE)
plt.plot(xex, RMSE, 'k')
#ax.axis([0., 100,10, 3000])
#xticks( range(1,35,4) )
plt.xlabel('Excedence [%]', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.savefig(os.path.join(fig_dir, 'Ah_exede_{}.png'.format(mod_nam)))

cmap = 'Greys'
norm = mpl.colors.Normalize(vmin=zh2_min * 100.0, vmax=zh2_max * 100.0)
X = np.arange(0, nx, 1)
Y = np.arange(0, ny, 1)
plt.ioff()
print('POD-DEIM reduced model error figures ... ')
for i in range(nts):
    plt.close()
    row_num = i
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    plt.title('POD-DEIM Reduced Model Error, Time: {} days \n'.format(row_num), fontsize=14)
    z_grid = (zh2[row_num, :].reshape(nx, ny)) * 100.0
    im = ax.pcolormesh(X, Y, z_grid, cmap=cmap, norm=norm)
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    ax.invert_yaxis()
    plt.xlabel('Column', fontsize=12)
    plt.ylabel('Row', fontsize=12)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label('Absolute Error in head (cm)', fontsize=12)
    plt.savefig(os.path.join(fig_dir, '0_err_t{0}.png'.format(i)), bbox_inches='tight')
print(' ... done')
plt.close('all')
norm = mpl.colors.Normalize(vmin=zA_min, vmax=zA_max)
i = 180
print('POD-DEIM reduced model error figures ... ')
for i in range(nts):
    plt.close()
    row_num = i
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect='equal')
    plt.title('POD-DEIM Reduced Model Error, Time: {} days \n'.format(row_num), fontsize=14)
    z_grid = (zA[row_num, :].reshape(nx, ny))
    im = ax.pcolormesh(X, Y, z_grid, cmap=cmap, norm=norm)
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    ax.invert_yaxis()
    plt.xlabel('Column', fontsize=12)
    plt.ylabel('Row', fontsize=12)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label('Nonlinear Error', fontsize=12)
    plt.savefig(os.path.join(fig_dir, 'Ah_err_t{0}.png'.format(i)))
print(' ... done')

norm = mpl.colors.Normalize(vmin=zh_min, vmax=zh_max)
for i in range(nts):
    row_num = i
    z_grid = zh[row_num, :].reshape(nx, ny)
    fig, ax = plt.subplots()
    im = ax.pcolormesh(X, Y, z_grid, cmap=cmap, norm=norm)
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    ax.invert_yaxis()
    cb = fig.colorbar(im, ax=ax)
    plt.savefig(os.path.join(fig_dir, '1_err_t{0}.png'.format(i)), bbox_inches='tight')
    plt.close()

# plot four at a time
print('POD-DEIM reduced model error figures ... ')
fig = plt.figure(figsize=(10, 10))
row_num = 64
norm = mpl.colors.Normalize(vmin=zh2_min * 100.0, vmax=zh2_max * 100.0)
ax = fig.add_subplot(2, 2, 1, aspect='equal')
ax.set_title('Time: {} days \n'.format(row_num+1), fontsize=10)
z_grid = (zh2[row_num, :].reshape(nx, ny)) * 100.0
im = ax.pcolormesh(X, Y, z_grid, cmap=cmap, norm=norm)
ax.axis([X.min(), X.max(), Y.min(), Y.max()])
ax.invert_yaxis()
#ax.set_xlabel('Column', fontsize=10)
ax.set_ylabel('Row', fontsize=10)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
#cb.set_label('Absolute Error in head (cm)', fontsize=10)

norm = mpl.colors.Normalize(vmin=zA_min, vmax=zA_max)
ax3 = fig.add_subplot(2, 2, 3, aspect='equal')
ax3.set_title('Time: {} days \n'.format(row_num+1), fontsize=10)
z_grid = (zA[row_num, :].reshape(nx, ny))
im = ax3.pcolormesh(X, Y, z_grid, cmap=cmap, norm=norm)
ax3.axis([X.min(), X.max(), Y.min(), Y.max()])
ax3.invert_yaxis()
ax3.set_xlabel('Column', fontsize=10)
ax3.set_ylabel('Row', fontsize=10)
cb = fig.colorbar(im, ax=ax3, shrink=0.8)
#cb.set_label('Nonlinear Error', fontsize=10)

row_num = 65
norm = mpl.colors.Normalize(vmin=zh2_min * 100.0, vmax=zh2_max * 100.0)
ax2 = fig.add_subplot(2, 2, 2, aspect='equal')
ax2.set_title('Time: {} days \n'.format(row_num+1), fontsize=10)
z_grid = (zh2[row_num, :].reshape(nx, ny)) * 100.0
im = ax2.pcolormesh(X, Y, z_grid, cmap=cmap, norm=norm)
ax2.axis([X.min(), X.max(), Y.min(), Y.max()])
ax2.invert_yaxis()
#ax2.set_xlabel('Column', fontsize=10)
#ax2.set_ylabel('Row', fontsize=10)
cb = fig.colorbar(im, ax=ax2, shrink=0.8)
cb.set_label('Absolute Error in head (cm)', fontsize=10)

norm = mpl.colors.Normalize(vmin=zA_min, vmax=zA_max)
ax4 = fig.add_subplot(2, 2, 4, aspect='equal')
ax4.set_title('Time: {} days \n'.format(row_num+1), fontsize=10)
z_grid = (zA[row_num, :].reshape(nx, ny))
im = ax4.pcolormesh(X, Y, z_grid, cmap=cmap, norm=norm)
ax4.axis([X.min(), X.max(), Y.min(), Y.max()])
ax4.invert_yaxis()
ax4.set_xlabel('Column', fontsize=10)
#ax4.set_ylabel('Row', fontsize=10)
cb = fig.colorbar(im, ax=ax4, shrink=0.8)
cb.set_label('Nonlinear Error', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, '4_err.png'), bbox_inches='tight')
print(' ... done')
plt.show()

plt.savefig(os.path.join(fig_dir, 'Ah_err_t{0}.png'.format(i)))


MAE = sum(zh2.T, 0) / float(nts)
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
plt.close()
plt.close('all')
