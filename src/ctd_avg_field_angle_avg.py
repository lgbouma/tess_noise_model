'''
Given the output of `generate_bigfrac.py`, compute a "representative" PRF, by
averaging over sub-pixel centroid offsets, and taking a "representative" field
angle. (It turns out, it is also OK to average over the wavelength dependence).

For points drawn uniformly over a unit square, their average distance from the
center is one sixth of the parabolic constant.

For TESS cameras, this means the typical separation of a star from the optical
axis is ~9.2deg.

So I will average the 6deg and 12deg PRFs as the "representative" field angle.

It turns out that the temperature dependence of the PRF is much less than the
field angle dependence. So I will also take a mean over the temperature.

The output is saved to ../results/average_PRF.fits
'''
from __future__ import print_function, division

import numpy as np, matplotlib.pyplot as plt
from astropy.io import fits

def plot_PRF_vs_temperature(PRF, temps):

    plt.close('all')
    f, ax = plt.subplots(figsize=(4,4))

    n_colors = len(temps)
    colors = plt.cm.YlOrRd(np.linspace(0.2,1,n_colors))

    for t_ind, temp in enumerate(temps):

        this_PRF = PRF[:, t_ind]

        CFF = np.cumsum(np.sort(this_PRF)[::-1]) # cumulative flux fraction

        ax.plot(list(range(len(CFF))),
                CFF,
                color=colors[t_ind],
                label='{:d}K'.format(temp)
               )

    ax.legend(loc='best')
    ax.set_xlabel('pixels in aperture')
    ax.set_ylabel('cumulative flux fraction, $'+r'\a'+'pprox 9^\circ$ field angle')
    ax.set_xscale('log')

    outname = '../results/PRF_vs_temperature.pdf'
    f.savefig(outname, bbox_inches='tight')

if __name__ == '__main__':

    prf_file = '../results/bigfrac.fits'
    hdu = fits.open(prf_file)
    big_frac = hdu[0].data

    # shape: 20, 10, N_field_angles, N_pix, N_filters
    shape = big_frac.shape

    # average over the sub-pixel centroid offsets (and 45 deg rotations)
    PRF_avg_over_ctd = np.sum(np.sum(big_frac, axis=0), axis=0)/(shape[0]*shape[1])

    # linear mean of the 6deg and 12deg field angles, to get something like the
    # representative 9 deg field angle.
    PRF_avg_over_fa_ctd = (PRF_avg_over_ctd[1,:,:] + PRF_avg_over_ctd[2,:,:])/2

    #'F0 (7200)','F5 (6440)','G0 (6030)','G5 (5770)','K0 (5250)','K5 (4350)',
    #'M1 (3600)','M3 (3250)','M5 (2800)'
    temperatures = [7200, 6440, 6030, 5770, 5250, 4350, 3600, 3250, 2800]

    norm_PRF = np.zeros_like(PRF_avg_over_fa_ctd)
    PRF = PRF_avg_over_fa_ctd
    # make sure PRF is appropriately normalized
    for t_ind, temp in enumerate(temperatures):
        norm_PRF[:, t_ind] = PRF[:, t_ind] / np.sum(PRF[:, t_ind])

    plot_PRF_vs_temperature(norm_PRF, temperatures)

    # the result of the above plotting routine is that it's OK to average over
    # temperature.
    nPRF_avg_over_fa_ctd_temp = np.mean(norm_PRF,axis=1)

    outname = 'average_PRF.fits'
    hdu = fits.PrimaryHDU()
    hdu.data = nPRF_avg_over_fa_ctd_temp
    hdu.writeto('../results/{:s}'.format(outname), overwrite=True)
