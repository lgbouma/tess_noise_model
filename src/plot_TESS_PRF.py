'''
Plot TESS pixel response function (PRF) after sorting and summing to show the
cumulative fraction of light collected for a given number of pixels in the
photometric aperture. We show this fraction for three field angles and three
values of stellar effective temperature. The dotted line is for T eff = 3000 K,
the solid line is for 5000 K, and the dashed line is for 7000 K. These
temperatures span most of the range of the TESS target stars.

(This is Fig 13 of Sullivan et al, given modified PSFs, as well as actual
corrections to the PSF->PRF integration code)
'''

import numpy as np, matplotlib.pyplot as plt
from astropy.io import fits

quick = True

outname = 'bigfrac.fits' if not quick else 'bigfrac_quick.fits'
hdu = fits.open('../results/'+outname)

# `big_frac` is a numpy array of the PRF values, as a function of pixel index,
# wavelength filter, field angle, and a randomly selected (x,y) pair that
# spatially shifts the PRF.
#big_frac = np.zeros((20, 10, N_field_angles, N_pix, N_filters))

big_frac = hdu[0].data
shape = big_frac.shape

PRF_avg_over_ctd = np.sum(np.sum(big_frac, axis=0), axis=0)/(shape[0]*shape[1])

#'F0 (7200)','F5 (6440)','G0 (6030)','G5 (5770)','K0 (5250)','K5 (4350)',
#'M1 (3600)','M3 (3250)','M5 (2800)'
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

field_angles = np.array([0, 6, 12, 17])
linestyles = [':','-','--']

plt.close('all')
f,ax = plt.subplots(figsize=(4,4))

quick = True
if not quick:
    for fa_ind, field_angle in enumerate(field_angles):

        # 7200, 5250, 2800
        for ls_ind, temp_ind in enumerate([0,4,8]):

            this_PRF = np.sort(PRF_avg_over_ctd[fa_ind, :, temp_ind])[::-1]

            cumulative_flux_fraction = np.cumsum(this_PRF)

            # ensure appropriate normalization...
            #cumulative_flux_fraction /= np.max(cumulative_flux_fraction)

            if ls_ind == 1:
                ax.plot(list(range(len(cumulative_flux_fraction))),
                        cumulative_flux_fraction,
                        color=colors[fa_ind],
                        linestyle=linestyles[ls_ind],
                        label='${:d}^\circ$'.format(field_angle)
                       )
            else:
                ax.plot(list(range(len(cumulative_flux_fraction))),
                        cumulative_flux_fraction,
                        color=colors[fa_ind],
                        linestyle=linestyles[ls_ind]
                       )

else:
    fa_ind = 0
    ls_ind = 0
    temp_ind = 4
    this_PRF = np.sort(PRF_avg_over_ctd[fa_ind, :, temp_ind])[::-1]

    cumulative_flux_fraction = np.cumsum(this_PRF)

    # ensure appropriate normalization...
    cumulative_flux_fraction /= np.max(cumulative_flux_fraction)

    ax.plot(list(range(len(cumulative_flux_fraction))),
            cumulative_flux_fraction,
            color=colors[fa_ind],
            linestyle=linestyles[ls_ind],
            label='$0^\circ$'
           )


ax.legend(loc='best')
ax.set_xlabel('pixels in aperture')
ax.set_ylabel('cumulative flux fraction')
ax.set_xscale('log')

outname = '../results/PRF_as_CFF.pdf' if not quick \
          else '../results/PRF_as_CFF_quick.pdf'
f.savefig(outname, bbox_inches='tight')
