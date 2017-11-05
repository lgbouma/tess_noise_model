'''
make some plots of the supersampled TESS PSFs
'''

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import h5py

psf_dir = '../data/PSFs_from_Deb_sent_May_17_2017/'
psf_file = 'CamSN06_CCD1_16May2017_PSFnorm_bfd3p346focus_+0.03.mat'

f = h5py.File(psf_dir+psf_file, 'r')

image_size = f[f.get('PSF_stellar/PSF')[0,0]].shape
image_len = image_size[0]

X, Y = np.meshgrid(range(image_len),range(image_len))

filt_i = 8
ang_i = 3

filts_angs = [(8,3), (8,2), (8,1), (8,0),
              (4,3), (4,2), (4,1), (4,0),
              (0,3), (0,2), (0,1), (0,0)]

for filt_i, ang_i in filts_angs:
    print(filt_i, ang_i)

    this_dat = np.array(f[f.get('PSF_stellar/PSF')[filt_i,ang_i]])

    plt.close('all')

    fig,ax = plt.subplots(figsize=(4.8,4))

    ais = ax.pcolor(X, Y, this_dat, cmap='gray')

    fig.colorbar(ais, ax=ax, label='psf value, supersampled')

    fig.savefig(
        '../results/supersampled_psf_filt{:d}_ang{:d}.png'.format(
            filt_i, ang_i),
        dpi=300)

    del ais
    del this_dat
