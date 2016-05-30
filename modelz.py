__author__ = 'stanko'


class Zmode(object):
    def __init__(self, nrow, ncol, nlay, nsp, nss, dsm, simdir, indir, outdir, modnam, nrv=0, ndeim=0, drm=0):
        self.nrow = nrow
        self.ncol = ncol
        self.nlay = nlay
        self.nsp = nsp  # number of stress periods
        self.nss = nss  # number of snapshots
        self.dsm = dsm  # dimension of the system matrix
        self.nrv = nrv  # number of reduced variables
        self.drm = drm  # reduced model dimension
        self.ndeim = ndeim  # number of DEIM rows retained
        self.simdir = simdir
        self.indir = indir
        self.outdir = outdir
        self.modnam = modnam
        self.err = 0.01  # replace with error class

    def __repr__(self):
        info1 = 'problem {0} has {1} columns, {2} rows , and {3} layers \n'.format(self, self.nrow, self.ncol, self.nlay)
        info2 = 'there are {0} stress periods, and {1} snapshots, resulting in a {2} system matrix \n'.format(self.nsp, self.nss, self.dsm)
        info3 = 'there are {0} reduced variables and {1} deim rows, resulting in a {2} reduced model \n'.format(self.nrv, self.deim, self.drm)
        info4 = 'the directory for the simulation model is {0} \n'.format(self.simdir)
        return info1 + info2 + info3 + info4 + '\n'
