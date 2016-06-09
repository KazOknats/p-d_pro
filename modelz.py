from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
__author__ = 'stanko'


class Zmode(object):
    def __init__(self, nrow, ncol, nlay, nsp, nts, nss, dsm, mod_nam, mod_type, nrv=0, ndeim=0, drm=0):
        self.nrow = nrow
        self.ncol = ncol
        self.nlay = nlay
        self.nsp = nsp  # number of stress periods
        self.nts = nts  # number of time steps
        self.nss = nss  # number of snapshots
        self.dsm = dsm  # dimension of the system matrix
        self.nrv = nrv  # number of reduced variables
        self.drm = drm  # reduced model dimension
        self.ndeim = ndeim  # number of DEIM rows retained
        self.mod_nam = mod_nam
        self.mod_type = mod_type
        self.sim_dir = os.path.join('..', 'model', 'exe')
        self.res_dir = os.path.join(self.sim_dir, '_result')
        self.mo_dir = os.path.join(self.res_dir, mod_type)
        self.fig_dir = os.path.join(self.sim_dir, '_fig')
        self.err = 0.01  # replace with error class
        if not os.path.exists(self.fig_dir):
            os.mkdir(self.fig_dir)
        pdf_dir = os.path.join(self.fig_dir, '_pdf')
        if not os.path.exists(pdf_dir):
            os.mkdir(pdf_dir)

    def __repr__(self):
        info1 = 'problem {0} has {1} columns, {2} rows , and {3} layers \n'.format(self, self.nrow, self.ncol, self.nlay)
        info2 = 'there are {0} stress periods, and {1} snapshots, resulting in a {2} system matrix \n'.format(self.nsp, self.nss, self.dsm)
        info3 = 'there are {0} reduced variables and {1} deim rows, resulting in a {2} reduced model \n'.format(self.nrv, self.deim, self.drm)
        info4 = 'the directory for the simulation model is {0} \n'.format(self.simdir)
        return info1 + info2 + info3 + info4 + '\n'


class Zrez(object):
    def __init__(self, nts, nx, ny, ss_h, ss_hr):
        zer = np.abs(ss_h - ss_hr)
        self.err = zer
        self.hr_min = ss_hr.min()
        self.h_min = ss_h.min()
        self.h_max = ss_h.max()
        self.e_min, self.e_max = np.abs(zer).min(), np.abs(zer).max()
        self.emx_loc = np.unravel_index(np.argmax(zer), zer.shape)
        self.xy_loc = np.unravel_index(np.argmax(zer), (nts, nx, ny))
        self.e_n = zer.size
        self.e_shape = zer.shape
        self.rmse = np.linalg.norm(zer) / np.sqrt(self.e_n)
        self.rmse_xy = np.linalg.norm(zer, axis=0) / np.sqrt(self.e_n)
        self.rmse_t = np.linalg.norm(zer, axis=1) / np.sqrt(self.e_n)
        self.rmse_avg = np.mean(self.rmse)
        self.nrmse = self.rmse/self.e_max
        self.nrmse_xy = self.rmse_xy/self.e_max
        self.nrmse_t = self.rmse_t/self.e_max

    def __repr__(self):
        info1 = '*** error in h ***\n'
        info2 = ' min head (full): {}\n'.format(self.h_min)
        info3 = ' min head (red): {}\n'.format(self.hr_min)
        info4 = ' size: {}\n shape: {}\n'.format(self.e_n, self.e_shape)
        info5 = ' min, max: {}, {}\n'.format(self.e_min, self.e_max)
        info6 = ' max loc (t, rowcol): {}\n'.format(self.emx_loc)
        info7 = ' max loc (t,row,col): {}\n'.format( self.xy_loc)
        info8 = ' AVG RMSE: {}\n NRMSE: {}'.format(self.rmse_avg, self.nrmse)
        return info1 + info2 + info3 + info4 + info5 + info6 + info7 + info8 + '\n'


def plot_2d_nrmse(nam, typ, x, y, z, sx, sy, sf):
    nrm = mpl.colors.Normalize(vmin=z.min(), vmax=z.max())
    print('plot {} NRMSE for head over all time for each cell ... '.format(typ))
    zgrid = (z.reshape(nx, ny))
    figf = plt.figure(figsize=(sx, sy))
    axf = figf.add_subplot(1, 1, 1, aspect='equal')
    axf.set_title('{} Reduced Model Error \n'.format(typ), fontsize=sf)
    imf = axf.pcolormesh(x, y, zgrid, cmap=cmap, norm=nrm)
    axf.axis([x.min(), x.max(), y.min(), y.max()])
    axf.invert_yaxis()
    axf.set_xlabel('Column', fontsize=sf)
    axf.set_ylabel('Row', fontsize=sf)
    cbf = figf.colorbar(imf, ax=axf, shrink=0.9)
    cbf.set_label('NRMSE', fontsize=sf)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '2d_err_{}_{}.png'.format(nam, typ)), bbox_inches='tight')
    plt.savefig(os.path.join(pdf_dir, '2d_err_{}_{}.pdf'.format(nam, typ)), bbox_inches='tight')
    print(' ... done \n')
    plt.close()