from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import flopy
import flopy.utils.binaryfile as bf
import modelz as zm
__author__ = 'stanko'


# set directory structure
w_dir = os.getcwd()
model_nam = "2D_05"
sim_dir = os.path.join('..', 'model', 'exe')
model_dir = os.path.join('..', 'model', model_nam)
namfile = os.path.join(model_dir, '{}.nam'.format(model_nam))
exe_name = os.path.join(sim_dir, 'swt_v4x64')
in_dir = os.path.join(model_dir, 'raw_input')
out_dir = os.path.join(model_dir, 'raw_output')
fig_dir = os.path.join(w_dir, 'fig_tab')

if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)

# define model instances of Zmode class
zmf = zm.Zmode(nrow=49, ncol=49, nlay=1, nsp=65, nss=60, dsm=2401,
               simdir=sim_dir, indir=in_dir, outdir=out_dir, modnam=model_nam)

# set plotting parameters
plt.rcdefaults()  # reset to default
plt.style.use('ggplot')
dpi = 150
newparams = {'figure.dpi': dpi, 'savefig.dpi': dpi,
             'font.size': 12, 'legend.fontsize': 12, 'axes.labelsize': 12,
             'xtick.labelsize': 12, 'ytick.labelsize': 12,
             'pdf.fonttype': 42, 'pdf.compression': 0}
plt.rcParams.update(newparams)

# turn off interactive mode
plt.ioff()

# import model
mf = flopy.modflow.Modflow.load(namfile, exe_name=exe_name, model_ws=model_dir)
pnf = flopy.utils.mfreadnam.parsenamefile(namfile, mf.mfnam_packages)

# prepare for plotting
dis = flopy.modflow.ModflowDis.load(os.path.join(in_dir, model_nam+'.dis'), mf)
mf.dis.plot(filename_base="dis_fig", file_extension='png', mflay=21)
bas = flopy.modflow.ModflowBas.load(os.path.join(in_dir, model_nam+'.bas'), mf, ext_unit_dict=pnf)

# make figure
print('making model grid figures ... ')
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect='equal')
modelmap = flopy.plot.ModelMap(sr=mf.dis.sr)
linecollection = modelmap.plot_grid(colors='black', alpha=0.5)
modelmap = flopy.plot.ModelMap(model=mf)
quadmesh = modelmap.plot_ibound()
quadmesh = modelmap.plot_bc('CHD')
quadmesh = modelmap.plot_bc('GHB')
quadmesh = modelmap.plot_bc('WEL')
quadmesh = modelmap.plot_bc('RIV')
quadmesh = modelmap.plot_bc('DRN')
# plt.show()
# save figure
fignam = model_nam+'model_grid_05'
plt.savefig(os.path.join(fig_dir, fignam+'.png'))
plt.savefig(os.path.join(fig_dir, fignam+'.pdf'))
print(' ... done')
# close figure
plt.close('all')

