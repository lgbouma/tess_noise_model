'''
At T=5250K, how do the cumulative flux fractions used by Sullivan+ 2015,
Bouma+ 2017, and now to be used in the super-official TSWG TESS prioritization
metric's noise model compare against e/other?

NB. Nit-picking, S+15's and B+17's temperatures were not exactly 5250K (they
were within a few hundred deg K of it).
'''
import numpy as np, matplotlib.pyplot as plt
from astropy.io import fits

#XXX set this to be 0, 1, 2, or 3.

newest = '../results/bigfrac.fits'
bouma = '../data/dfrac_asbuilt_75c_0f.fits'
sullivan = '../data/dfrac_t75_f3p31_3.fits'
prf_files = [newest, bouma, sullivan]

field_angles = np.array([0, 6, 12, 17])
linestyles = [':','-','--']

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

for fa_ind in [0,1,2,3]:
    print(fa_ind)

    plt.close('all')
    f,ax = plt.subplots(figsize=(4,4))

    for name, prf_file in zip(['this work','B+17','S+15'], prf_files):

        # `big_frac` is a numpy array of the PRF values, as a function of pixel index,
        # wavelength filter, field angle, and a randomly selected (x,y) pair that
        # spatially shifts the PRF.
        #big_frac = np.zeros((20, 10, N_field_angles, N_pix, N_filters))
        hdu = fits.open(prf_file)
        big_frac = hdu[0].data

        if name=='B+17' or name=='S+15':
            big_frac = big_frac.T

        shape = big_frac.shape

        PRF_avg_over_ctd = np.sum(np.sum(big_frac, axis=0), axis=0)/(shape[0]*shape[1])

        #'F0 (7200)','F5 (6440)','G0 (6030)','G5 (5770)','K0 (5250)','K5 (4350)',
        #'M1 (3600)','M3 (3250)','M5 (2800)'

        ls_ind = 1
        temp_ind = 4
        this_PRF = np.sort(PRF_avg_over_ctd[fa_ind, :, temp_ind])[::-1]

        cumulative_flux_fraction = np.cumsum(this_PRF)

        # ensure appropriate normalization...
        cumulative_flux_fraction /= np.max(cumulative_flux_fraction)

        ax.plot(list(range(len(cumulative_flux_fraction))),
                cumulative_flux_fraction,
                linestyle=linestyles[ls_ind],
                label='${:d}^\circ$, {:s}'.format(field_angles[fa_ind], name)
               )


    ax.legend(loc='best')
    ax.set_xlabel('pixels in aperture')
    ax.set_ylabel('cumulative flux fraction')
    ax.set_xscale('log')

    outname = '../results/PRF_comparison_fa{:d}.pdf'.format(fa_ind)
    f.savefig(outname, bbox_inches='tight')
