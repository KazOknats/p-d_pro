from __future__ import print_function
import os
import matplotlib.pyplot as plt
import flopy
import numpy as np
# import modelz as zm
import flopy.utils.formattedfile as ff
__author__ = 'stanko'

w_dir = os.getcwd()
model_nam = "1D_02"
sim_dir = os.path.join('..', 'model', model_nam)
namfile = os.path.join(sim_dir, '{}.nam'.format(model_nam))
exe_name = os.path.join(sim_dir, 'swt_v4x64')
in_dir = os.path.join(sim_dir, 'raw_input')
out_dir = os.path.join(sim_dir, 'raw_output')
fig_dir = os.path.join(sim_dir, 'fig')
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)

# define model instances of Zmode class
# zmf = zm.Zmode(nrow=1, ncol=200, nlay=1, nsp=90, nss=90, dsm=200,
#                simdir=sim_dir, indir=in_dir, outdir=out_dir, modnam=model_nam)

# set plotting parameters
plt.rcdefaults()  # reset to default
# plt.style.use('ggplot')
dpi = 300
newparams = {'figure.dpi': dpi, 'savefig.dpi': dpi,
             'font.size': 12, 'legend.fontsize': 12, 'axes.labelsize': 12,
             'xtick.labelsize': 12, 'ytick.labelsize': 12,
             'pdf.fonttype': 42, 'pdf.compression': 0}
plt.rcParams.update(newparams)

# turn off interactive mode
plt.ioff()

# import model
print('loading model with flopy ... ')
mf = flopy.modflow.Modflow.load(namfile)
print(' ... done')

mf.dis.plot(filename_base="dis_fig", file_extension='png')
plt.show(fpdp)
plt.clf()

xmin = 0
xmax = 2000
ymin = 0
ymax = 50
v = [xmin, xmax, ymin, ymax]

font = {'family': 'serif',
        'color':  'navy',
        'weight': 'semibold',
        'size': 14,
        'backgroundcolor': 'white',
        }

fhd_file = os.path.join(sim_dir, model_nam + '.fhd')
hdobj = ff.FormattedHeadFile(fhd_file, precision='single', verbose=True)
hdobj.list_records()
rec = hdobj.get_data(idx=59)
levels = np.arange(-50, 0, 2)

fhd_file = os.path.join(sim_dir, '1D_01.fhd')
hdobj1 = ff.FormattedHeadFile(fhd_file, precision='single', verbose=True)
rec1 = hdobj1.get_data(idx=59)

#  zn = mf.lpf.hk.plot(masked_values=[0.], colorbar=True)

fig = plt.figure(figsize=(10, 6))

ax = fig.add_subplot(2, 1, 1)
xs = flopy.plot.ModelCrossSection(model=mf, line={'row': 0})
csa = xs.plot_array(rec, masked_values=[999.], head=rec, alpha=0.5)

ax = fig.add_subplot(2, 1, 2)
xs = flopy.plot.ModelCrossSection(model=mf, line={'row': 0})
# ct = xs.contour_array(rec, masked_values=[999.], head=rec, levels=levels)
# patches = xs.plot_ibound(head=rec)
zn = xs.plot_array(mf.lpf.hk, cmap='YlOrRd', alpha=0.5)
wt = xs.plot_surface(rec, masked_values=[999.], color='blue', lw=2)
linecollection = xs.plot_grid()

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 1, 1)
xs = flopy.plot.ModelCrossSection(model=mf, line={'row': 0})
zn = xs.plot_array(mf.lpf.hk, cmap='YlOrRd', alpha=0.25)
wt = xs.plot_surface(rec, masked_values=[999.], color='blue', lw=2, label='Q = 200')
wt = xs.plot_surface(rec1, masked_values=[999.], color='black', lw=2, ls='--', label='Q = 100')
linecollection = xs.plot_grid(alpha=0.5)
plt.xlabel('Horizontal Distance (m)', fontsize=12)
plt.ylabel('Hydraulic Head (m)', fontsize=12)
ax.legend(loc='lower left')
plt.text(450., -5., r'$K_x = 0.4$', fontdict=font)
plt.text(850., -5., r'$K_x = 1.2$', fontdict=font)
fignam = model_nam+'_redmod'
plt.savefig(os.path.join(fig_dir, fignam+'.png'), bbox_inches='tight')
plt.savefig(os.path.join(fig_dir, fignam+'.pdf'), bbox_inches='tight')
plt.close()

# make figure
print('making model grid figures ... ')
fig = plt.figure(figsize=(8, 3))
ax = fig.add_subplot(1, 1, 1, aspect='auto')
plt.ylim(ymin, ymax)
plt.xlim(xmin, xmax)
modelmap = flopy.plot.ModelMap(model=mf)
linecollection = modelmap.plot_grid(colors='black', alpha=0.5)
ax.set_aspect('auto')
quadmesh = modelmap.plot_ibound()
quadmesh = modelmap.plot_bc('CHD')
quadmesh = modelmap.plot_bc('WEL')
flopy.contour_array(a, masked_values=None, head=None, **kwargs)
# plt.show()
# save figure
fignam = model_nam+'model_grid'
plt.savefig(os.path.join(fig_dir, fignam+'.png'))
plt.savefig(os.path.join(fig_dir, fignam+'.pdf'))
print(' ... done')
plt.clf()
# close figure
plt.close('all')

