'''
Updated parametrized noise model

Given source mag, an effective temperature, and a coordinate, this function
gives predicted TESS RMS for the source.

It does so by computing the optimal aperture, for a fixed "representative"
field angle, using the updated PSF.

The relevant function is `noise_model(...)`.

If run on the command line,

    >>> $(bash) python noise_model.py

then this script produces a csv file with the tabulated values.

NOTE:
This code is derivative of both Zach Berta-Thompson's SNR calculator and Josh
Winn's IDL TESS SNR calculator.

The former is at https://github.com/zkbt/spyffi/blob/master/Noise.py.
The latter is saved in this directory (`JNW_calc_noise.pro`).

Author: Luke Bouma.
Date: Sat 04 Nov 2017 08:40:37 AM EDT
'''
from __future__ import division, print_function

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as units

###############################################################################
# Fixed TESS properties are kept as globals.

global subexptime, e_pix_ro, effective_area, sys_limit, pix_scale

subexptime = 2.0      # subexposure time [seconds] (n_exp = exptime/subexptime)
e_pix_ro = 10.0       # rms in no. photons/pixel from readout noise
effective_area = 73.0 # geometric collecting area
sys_limit = 60.0      # minimum uncertainty in 1 hr of data, in ppm
pix_scale = 21.1      # arcsec per pixel

###############################################################################

def photon_flux_from_source(Cousins_I, T_eff):
    '''
    in:
        Cousins_I (float): of the source
        T_eff (float): of the source

    out:
        photon flux from the source in the TESS band [units: ph/s/cm^2].
    '''

    # Equation on page 3 of Josh Winn's photon fluxes memo. Fit of photon
    # fluxes as a linear function of T_eff.

    # Uses the TESS spectral response function as predicted in 2010 (though
    # there are new lab measurements from Akshata). Gives photon flux in units
    # of ph/s/cm^2, for an I=0 source. This means:
    # response function = lens throughput * CCD quantum efficiency.

    # So really, this is equivalent to the electron flux.

    Gamma_T = ( 1.6301 + (2.9468*10**(-5)) * (T_eff - 5000) )*1e6

    return Gamma_T * 10**(-0.4 * Cousins_I)


def get_sky_bkgnd(coords, exptime):
    '''
    in:
        input coordinate (astropy SkyCoord instance)

        exposure time (seconds)

    out:
        sky background (zodiacal + bkgnd at coord) [units: e/px]
    '''

    elat = coords.barycentrictrueecliptic.lat.value
    elon = coords.barycentrictrueecliptic.lon.value
    glat = coords.galactic.b.value
    glon = coords.galactic.l.value

    # Solid area of a pixel (arcsec^2).
    omega_pix = pix_scale ** 2.

    # Photoelectrons/pixel from zodiacal light.
    dlat = (np.abs(elat) - 90.) / 90.
    vmag_zodi = 23.345 - 1.148 * dlat ** 2.

    # Eqn (3) from Josh Winn's memo on sky backgrounds. This comes from
    # integrating a model ZL spectrum over the TESS bandpass.
    e_pix_zodi = 10.0 ** (-0.4 * (vmag_zodi - 22.8)) * 2.39e-3 * \
                                    effective_area * omega_pix * exptime

    # Photoelectrons/pixel from background star fluid.
    dlat = np.abs(glat) / 40.0
    dlon = glon
    q = (dlon > 180.)
    dlon[q] = 360. - dlon[q]
    dlon = np.abs(dlon) / 180.0
    p = [18.9733, 8.833, 4.007, 0.805]
    imag_bgstars = p[0] + p[1] * dlat + p[2] * dlon ** (p[3])

    # Eqn (7) (fit to Fig 7) of Josh Winn's memo on sky backgrounds. This is a
    # simple linear fit to the star counts from the Besancon model. It was
    # sanity-checked against a few sight-line in the USNO-B1.0 catalog.
    e_pix_bgstars = 10.0 ** (-0.4 * imag_bgstars) * 1.7e6 * \
                                    effective_area * omega_pix * exptime

    return e_pix_zodi + e_pix_bgstars


def noise_model(
        I_mag,
        T_eff,
        coords=np.array((42,42)),
        exptime=120):
    '''
    ----------
    Mandatory inputs:

    either all floats, or else all 1d numpy arrays of length N_sources.

        I_mag:
            Cousins I magnitude of the source(s)

        T_eff:
            effective temperature of the source(s). [units: K]

    ----------
    Optional inputs:

        coords:
            target coordinates, a (N_sources * 2) numpy array of (ra, dec),
            specified in degrees.

        exptime (float):
            total exposure time in seconds. Must be a multiple of 2 seconds.

    ----------
    Returns:

        [N_sources x 6] array of:
            optimal number of pixels,
            noise for optimal number of pixels,
            each of the noise components (star, sky, readout, systematic).

    '''

    # Check inputs. Convert coordinates to astropy SkyCoord instance.
    if not isinstance(I_mag, np.ndarray):
        I_mag = np.array([I_mag])
    if not isinstance(T_eff, np.ndarray):
        T_eff = np.array([T_eff])
    assert isinstance(coords, np.ndarray)
    if len(coords.shape)==1:
        coords = coords.reshape((1,2))
    assert coords.shape[1] == 2

    coords = SkyCoord(
                 ra=coords[:,0]*units.degree,
                 dec=coords[:,1]*units.degree,
                 frame='icrs'
                 )

    assert exptime % subexptime == 0, \
            'Exposure time must be multiple of 2 seconds.'
    assert I_mag.shape[0] == T_eff.shape[0] == coords.shape[0]

    # Basic quantities.
    N_sources = len(I_mag)
    N_exposures = exptime/subexptime

    # Photon flux from source in ph/s/cm^2.
    f_ph_source = np.array(photon_flux_from_source(I_mag, T_eff))

    # Compute number of photons from source, per exposure.
    ph_source = f_ph_source * effective_area * exptime

    # Load in average PRF produced by `ctd_avg_field_angle_avg.py`.
    prf_file = '../results/average_PRF.fits'
    hdu = fits.open(prf_file)
    avg_PRF = hdu[0].data

    # Compute cumulative flux fraction, sort s.t. the brightest pixel is first.
    CFF = np.cumsum(np.sort(avg_PRF)[::-1])

    # For each source, compute the number of photons collected (in each
    # exposure) as a function of aperture size. Save as array of [N_sources *
    # N_pixels_in_aperture].
    ph_source_all_ap = ph_source[:, None] * CFF[None, :]

    # Convert to number of electrons collected as a function of aperture size.
    # These are the same, since Josh Winn's photon flux formula already
    # accounts for the quantum efficiency.
    e_star_all_ap = ph_source_all_ap

    e_sky = get_sky_bkgnd(coords, exptime)

    # Array of possible aperture sizes: [1,2,...,max_N_ap]
    N_pix_aper = np.array(range(1,len(CFF)+1))

    e_sky_all_ap = e_sky[:, None] * N_pix_aper[None, :]

    #########################################################################
    # Find the optimal aperture by computing noises for all aperture sizes, #
    # then taking the minimum.                                              #
    #########################################################################
    noise_star_all_ap = np.sqrt(e_star_all_ap) / e_star_all_ap

    noise_sky_all_ap = np.sqrt(N_pix_aper * e_sky_all_ap) / e_star_all_ap

    noise_ro_all_ap = np.sqrt(N_pix_aper * N_exposures) * e_pix_ro / e_star_all_ap

    noise_sys_all_ap = np.zeros_like(e_star_all_ap) \
                       + sys_limit / 1e6 / np.sqrt(exptime / 3600.)

    noise_all_ap = np.sqrt(noise_star_all_ap ** 2. +
                           noise_sky_all_ap ** 2. +
                           noise_ro_all_ap ** 2. +
                           noise_sys_all_ap ** 2.)

    # Optimal number of pixels in aperture (must be at least 3).
    opt_inds = np.argmin(noise_all_ap, axis=1)
    N_pix_opt = N_pix_aper[opt_inds]
    N_pix_opt = np.maximum(3*np.ones_like(N_pix_opt), N_pix_opt)

    # Report the optimal noise, and optimal number of pixels.
    e_star_opt_ap = []
    e_sky_opt_ap = []
    for ix, opt_ind in enumerate(opt_inds):
        e_star_opt_ap.append(e_star_all_ap[ix,opt_ind])
        e_sky_opt_ap.append(e_sky_all_ap[ix,opt_ind])
    e_star_opt_ap = np.array(e_star_opt_ap)
    e_sky_opt_ap = np.array(e_sky_opt_ap)

    noise_star_opt_ap = np.sqrt(e_star_opt_ap) / e_star_opt_ap

    noise_sky_opt_ap = np.sqrt(N_pix_opt * e_sky_opt_ap) / e_star_opt_ap

    noise_ro_opt_ap = np.sqrt(N_pix_opt * N_exposures) * e_pix_ro / e_star_opt_ap

    noise_sys_opt_ap = np.zeros_like(e_star_opt_ap) \
                       + sys_limit / 1e6 / np.sqrt(exptime / 3600.)

    noise_opt_ap = np.sqrt(noise_star_opt_ap ** 2. +
                           noise_sky_opt_ap ** 2. +
                           noise_ro_opt_ap ** 2. +
                           noise_sys_opt_ap ** 2.)

    return np.array(
            [N_pix_opt,
             noise_opt_ap,
             noise_star_opt_ap,
             noise_sky_opt_ap,
             noise_ro_opt_ap,
             noise_sys_opt_ap]
            )


if __name__ == '__main__':

    # Produce a csv file with tabulated values of the noise model.

    Imags = np.arange(4,16+0.05,0.05)
    Teffs = 5000 * np.ones_like(Imags)

    # RA, dec. (90, -66) is southern ecliptic pole
    good_coords = np.array([90*np.ones_like(Imags), -66*np.ones_like(Imags)]).T

    # Towards galactic center.
    bad_coords = np.array([266.25*np.ones_like(Imags), -28.94*np.ones_like(Imags)]).T

    for name, coords in zip(['good', 'bad'], [good_coords, bad_coords]):

        out = noise_model(Imags, Teffs, coords=coords, exptime=3600)

        df = pd.DataFrame({
                'N_pix':out[0,:],
                'I_mag':Imags,
                'noise':out[1,:],
                'noise_star':out[2,:],
                'noise_sky':out[3,:],
                'noise_ro':out[4,:],
                'noise_sys':out[5,:]
                })

        df.to_csv('../results/noise_model_{:s}_coords.csv'.format(name),
                index=False, float_format='%.4g')

