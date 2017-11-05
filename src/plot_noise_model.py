'''
RMS vs Imag
and
optimal number of pixels vs Imag
'''
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import matplotlib as mpl
mpl.use('pgf')
pgf_with_custom_preamble = {
    'pgf.texsystem': 'pdflatex', # xelatex is default; i don't have it
    'font.family': 'serif', # use serif/main font for text elements
    'text.usetex': True,    # use inline math for ticks
    'pgf.rcfonts': False,   # don't setup fonts from rc parameters
    'pgf.preamble': [
        '\\usepackage{amsmath}',
        '\\usepackage{amssymb}'
        ]
    }
mpl.rcParams.update(pgf_with_custom_preamble)

import numpy as np, pandas as pd, matplotlib.pyplot as plt

df_good = pd.read_csv('../results/noise_model_good_coords.csv')
df_bad = pd.read_csv('../results/noise_model_bad_coords.csv')

f, axs = plt.subplots(nrows=2,ncols=1,figsize=(4,6),sharex=True)

axs[0].plot(
        df_good['I_mag'],
        df_good['noise']*1e6,
        label='good coord, total',
        zorder=0
        )
axs[1].plot(
        df_good['I_mag'],
        df_good['N_pix'],
        label='good, $\equiv$ south ecliptic pole',
        zorder=0
        )

axs[0].plot(
        df_bad['I_mag'],
        df_bad['noise']*1e6,
        label='bad coord, total',
        linestyle='-',
        zorder=-1
        )
axs[1].plot(
        df_bad['I_mag'],
        df_bad['N_pix'],
        label='bad, $\equiv$ galactic center',
        linestyle='-',
        zorder=-1
        )

# Sullivan+ 2015 comparison
lnA = 3.29685004771
B = 0.850021465753
C = -0.285041632435
D = 0.0395908321369
E = -0.0022308015894
F = 4.73508403525e-5

I = np.array(df_good['I_mag'])
ln_sigma = lnA + B*I + C*I**2 + D*I**3 + E*I**4 + F*I**5
sigma_1hr = np.exp(ln_sigma)

axs[0].plot(
        df_good['I_mag'],
        sigma_1hr,
        label='S+15 total',
        linestyle='-',
        zorder=-2
        )

df = pd.read_csv(
        '../data/Sullivan_2015_optimalnumberofpixels.txt',
        comment='#', delimiter=','
        )
axs[1].plot(
        df['tmag'], #it's actually I_C
        df['npix'],
        label='S+15',
        linestyle='-',
        zorder=-2
        )

def N_pix(T):
    c_3 = -0.2592
    c_2 = 7.741
    c_1 = -77.792
    c_0 = 274.2989
    return c_3*T**3 + c_2*T**2 + c_1*T + c_0


axs[1].plot(df_good['I_mag'],
        np.maximum(3*np.ones_like(N_pix(df_good['I_mag'])),
            N_pix(df_good['I_mag'])),
        label='Stassun+17, p23',
        zorder=-3)


for substr in ['star','sky','ro','sys']:
    lsubstr = substr if substr != 'ro' else 'read'
    axs[0].plot(
            df_good['I_mag'],
            df_good['noise_'+substr]*1e6,
            label= lsubstr+' (good)',
            linestyle='--',
            lw=1
            )

axs[0].vlines(6.8, 25, 2.1e4, label='saturation', linestyle=':', lw=0.5)

axs[0].set_ylim([25,2.1e4])
axs[0].set_xlim([3.9, 16.1])
leg0 = axs[0].legend(loc='best', fontsize='x-small')
leg0.get_frame().set_alpha(1)
leg1 = axs[1].legend(loc='best', fontsize='x-small')
leg1.get_frame().set_alpha(1)
axs[1].set_xlabel('apparent Cousins I mag', fontsize='large')
axs[1].set_ylabel('pixels in optimal aperture', fontsize='large')
axs[0].set_ylabel('$\sigma\ [\mathrm{ppm}\ \mathrm{hr}^{-1/2}]$', fontsize='large')
axs[0].set_yscale('log')

for ax in axs:
    if ax == axs[1]:
        ax.xaxis.set_ticks_position('both')
    ax.tick_params(which='both', direction='in', zorder=0)

f.tight_layout(h_pad=-0.1)

outname = '../results/noise_model.pdf'
f.savefig(outname, bbox_inches='tight')
