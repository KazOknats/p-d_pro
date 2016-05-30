__author__ = 'stanko'
from string import letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import time
import csv
import os
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


start = time.time()
def elapsed():
    return time.time() - start

# mod_nam = '2D_unconfined_03'
mod_nam = '1D_unconfined_02'
fil_dir = os.path.join('..', 'model', mod_nam)

# import the snapshots of h
fil_nam = 'SnapShots_h.txt'
f = open(os.path.join(fil_dir, fil_nam), 'rb')
ss_h = np.genfromtxt(f, skip_header=1)


fil_nam = 'SnapShotsR.txt'
f = open(os.path.join(fil_dir, fil_nam), 'rb')
ss_hr = np.genfromtxt(f, skip_header=1)

Z = np.abs(ss_h - ss_hr)
Zlog10 = np.log10(Z)
z_min, z_max = np.abs(Z).min(), np.abs(Z).max()
nts = 90
t = 15
nx = 1
ny = 200
n = 20

X = np.arange(0, nx, 1)
Y = np.arange(0, ny, 1)
dx, dy = 1, 1
y, x = np.mgrid[slice(1, ny + dy, dy),
                slice(1, nx + dx, dx)]

levels = MaxNLocator(nbins=30).tick_values(z_min, z_max)
cmap = plt.get_cmap('PiYG')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
row_num = 25
i = 25

plt.ioff()


for i in range(nts):
    plt.pcolormesh(X, Y, Z[i, :].reshape(nx, ny), cmap=cmap, norm=norm, vmin=z_min, vmax=z_max)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.colorbar()
    plt.savefig(os.path.join(fil_dir, 'fig', '0_err_t{0}.png'.format(i)))
    plt.close()


plt.show()

fig, ax = plt.subplots()
im = ax.pcolormesh(X, Y, Z[:,nx*n:nx*n+nx], cmap=cmap, norm=norm, vmin=z_min, vmax=z_max)
plt.axis([X.min(), X.max(), Y.min(), Y.max()])
fig.colorbar(im, ax=ax)
plt.show()

X = np.arange(0, nx, 1)
Y = np.arange(0, ny, 1)
i = 30
for i in range(nts):
    row_num = i
    Z_grid = Z[row_num, :].reshape(nx, ny)
    fig, ax = plt.subplots()
    im = ax.pcolormesh(X, Y, Z_grid, cmap=cmap, norm=norm, vmin=z_min, vmax=z_max)
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    cb = fig.colorbar(im, ax=ax)
    plt.savefig(os.path.join(fil_dir, 'fig', '0_err_t{0}.png'.format(n)))

for n in range(49):
    fig, ax = plt.subplots()
    im = ax.pcolormesh(X, Y, Zlog10[:,nx*n:nx*n+nx], cmap=cm.RdBu,
                       vmin=abs(Zlog10).min(), vmax=abs(Zlog10).max(),
                       extent=[0, 49, 0, 65])
    cb = fig.colorbar(im, ax=ax)
    plt.savefig(os.path.join(fil_dir, 'fig', 'err_{0}.png'.format(n)))

plt.show()

fig, ax = plt.subplots()

p = ax.pcolor(X, Y, Z[:,nx*t:nx*t+nx], cmap=cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max())
cb = fig.colorbar(p, ax=ax)

ss_err[1:100, 1]
ss_logdif = np.log10(np.abs(ss_h - ss_Ah))
ss_logdif[1:100, 1]

# count data rows, to preallocate array

def count(f):
    while 1:
        block = f.read(65536)
        if not block:
             break
        yield block.count(',')

linecount = sum(count(f))
print '\n%.3fs: file has %s rows' % (elapsed(), linecount)

# pre-allocate array and load data into array
m = np.zeros(linecount, dtype=[('a', np.uint32), ('b', np.uint32)])
f.seek(0)
f = csv.reader(open('links.csv', 'rb'))
for i, row in enumerate(f):
    m[i] = int(row[0]), int(row[1])

print '%.3fs: loaded' % elapsed()
# sort in-place
m.sort(order='b')

print '%.3fs: sorted' % elapsed()

sns.set(style="white")

# read in data with numpy
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                 columns=list(letters[:26]))

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

alpha = 0.7
phi_ext = 2 * np.pi * 0.5

def flux_qubit_potential(phi_m, phi_p):
    return 2 + alpha - 2 * np.cos(phi_p) * np.cos(phi_m) - alpha * np.cos(phi_ext - 2*phi_p)


phi_m = np.linspace(0, 2*np.pi, 100)
phi_p = np.linspace(0, 2*np.pi, 100)
X,Y = np.meshgrid(phi_p, phi_m)
Z = flux_qubit_potential(X, Y).T

fig, ax = plt.subplots()

p = ax.pcolor(X/(2*np.pi), Y/(2*np.pi), Z, cmap=cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max())
cb = fig.colorbar(p, ax=ax)

fig, ax = plt.subplots()

im = ax.imshow(Z, cmap=cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), extent=[0, 1, 0, 1])
im.set_interpolation('bilinear')

cb = fig.colorbar(im, ax=ax)


######
# mod_nam = '2D_unconfined_03'
mod_nam = '1D_unconfined_02'
fil_dir = os.path.join('..', 'model', mod_nam)

# import the snapshots of h
fil_nam = 'SnapShots_h.txt'
f = open(os.path.join(fil_dir, fil_nam), 'rb')
ss_h = np.genfromtxt(f, skip_header=1)


fil_nam = 'SnapShots_Ah.txt'
f = open(os.path.join(fil_dir, fil_nam), 'rb')
ss_Ah = np.genfromtxt(f, skip_header=1)

Z = np.abs(ss_h - ss_Ah)
Zlog10 = np.log10(Z)
z_min, z_max = np.abs(Z).min(), np.abs(Z).max()
nts = 90
t = 15
nx = 100
ny = 100
n = 20

X = np.arange(0, nx, 1)
Y = np.arange(0, nts, 1)
dx, dy = 1, 1
y, x = np.mgrid[slice(1, ny + dy, dy),
                slice(1, nx + dx, dx)]

levels = MaxNLocator(nbins=30).tick_values(z_min, z_max)
cmap = plt.get_cmap('PiYG')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

fig, ax = plt.subplots()
im = ax.pcolormesh(X, Y, Z[:,nx*n:nx*n+nx], cmap=cmap, norm=norm, vmin=z_min, vmax=z_max)
plt.axis([X.min(), X.max(), Y.min(), Y.max()])
fig.colorbar(im, ax=ax)
plt.savefig(os.path.join(fil_dir, 'fig', '0_err_t{0}.png'.format(n)))

plt.show()

X = np.arange(0, nx, 1)
Y = np.arange(0, ny, 1)
i = 30
for i in range(nts):
    row_num = i
    Z_grid = Z[row_num, :].reshape(nx, ny)
    fig, ax = plt.subplots()
    im = ax.pcolormesh(X, Y, Z_grid, cmap=cmap, norm=norm, vmin=z_min, vmax=z_max)
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    cb = fig.colorbar(im, ax=ax)
    plt.savefig(os.path.join(fil_dir, 'fig', '0_err_t{0}.png'.format(i)))