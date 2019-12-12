import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import h5py
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--hdf5path', dest='hdf5path', type=str, default='./', help='path')
parser.add_argument('--savepath', dest='savepath', type=str, default='./', help='path')
args = parser.parse_args()

hdf5path = args.hdf5path
savepath = args.savepath
result_str = os.path.join(hdf5path, 'LearnedFwdSimPrediction.hdf5')

file_prediction = h5py.File(result_str, 'r')
dataset_CNN = file_prediction["result"]
dataset_HF = file_prediction["HF"]
dataset_LF = file_prediction["LF"]

origin = (0., 0.)
spacing = (7.5, 7.5)
tn = 1200.
shape = [401, 301]
nbpml = 40

font = {'family' : 'sans-serif',
        'size'   : 14}
import matplotlib
matplotlib.rc('font', **font)
Xstart = (shape[0] + 2 * nbpml) * spacing[0]
Tend = (shape[1] + 2 * nbpml) * spacing[1]

for iG in range(dataset_CNN.shape[3]):

    plt.figure()
    im = plt.imshow(np.transpose(dataset_HF[2, :, :, iG]), vmin=-.5, vmax=.5, cmap="Greys", \
        aspect='1',extent=[0,Xstart,Tend,0], interpolation="lanczos")
    plt.xlabel('Horizontal Location (m)')
    plt.ylabel('Depth (m)')
    plt.colorbar(im,fraction=0.038, pad=0.02)
    plt.grid(linestyle='--', linewidth=1, alpha=1, color='k')
    plt.savefig(os.path.join(savepath, 'wave-NonDispersed_' + str(iG) \
        + '.png'), format='png', bbox_inches='tight', dpi=100)

    plt.figure()
    im = plt.imshow(np.transpose(dataset_CNN[2, :, :, iG]), vmin=-.5, vmax=.5, cmap="Greys", \
        aspect='1',extent=[0,Xstart,Tend,0], interpolation="lanczos")
    plt.xlabel('Horizontal Location (m)')
    plt.ylabel('Depth (m)')
    plt.colorbar(im,fraction=0.038, pad=0.02)
    plt.grid(linestyle='--', linewidth=1, alpha=1, color='k')
    plt.savefig(os.path.join(savepath, 'wave-result_' + str(iG) \
        + '.png'), format='png', bbox_inches='tight', dpi=100)

    plt.figure()
    im = plt.imshow(np.transpose(dataset_LF[2, :, :, iG]), vmin=-.5, vmax=.5, cmap="Greys", \
        aspect='1',extent=[0,Xstart,Tend,0], interpolation="lanczos")
    plt.xlabel('Horizontal Location (m)')
    plt.ylabel('Depth (m)')
    plt.colorbar(im,fraction=0.038, pad=0.02)
    plt.grid(linestyle='--', linewidth=1, alpha=1, color='k')
    plt.savefig(os.path.join(savepath, 'wave-dispersed_' + str(iG) \
        + '.png'), format='png', bbox_inches='tight', dpi=100)

    plt.figure()
    im = plt.imshow(np.transpose(dataset_HF[2, :, :, iG] - dataset_CNN[2, :, :, iG]), vmin=-.5, vmax=.5, \
        cmap="Greys", aspect='1',extent=[0,Xstart,Tend,0], interpolation="lanczos")
    plt.xlabel('Horizontal Location (m)')
    plt.ylabel('Depth (m)')
    plt.colorbar(im,fraction=0.038, pad=0.02)
    plt.grid(linestyle='--', linewidth=1, alpha=1, color='k')
    plt.savefig(os.path.join(savepath, 'wave-error_' + str(iG) \
        + '.png'), format='png', bbox_inches='tight', dpi=100)

