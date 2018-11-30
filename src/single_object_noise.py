from __future__ import division, print_function

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as units

from noise_model import noise_model

if __name__ == '__main__':

    # Produce a csv file with tabulated values of the noise model.
    T_mag = np.array([11.778])

    # from TIC
    coords = np.array([353.562839, -42.061409]).T

    out = noise_model(T_mag, coords=coords, exptime=3600)

    df = pd.DataFrame({
            'N_pix':out[0,:],
            'T_mag':T_mag,
            'noise':out[1,:]*1e6,
            'noise_star':out[2,:]*1e6,
            'noise_sky':out[3,:]*1e6,
            'noise_ro':out[4,:]*1e6,
            'noise_sys':out[5,:]*1e6
            })

    print(df)
