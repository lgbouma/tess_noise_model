'''
This script dithers the TESS PSFs, and integrates them up to TESS pixels.

This is a python rewrite of `bigfracgen.m`, a matlab script written by Peter
Sullivan to do the same thing.

The output is `bigfrac.fits`, a file that contains a numpy array of the PRF
values, as a function of pixel index, wavelength filter, field angle, and a
sub-pixel centroid offsets.

Author: Luke Bouma.
Date: Thu 02 Nov 2017 08:11:52 AM EDT
'''
from __future__ import division, print_function

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import h5py
from scipy.interpolate import interp2d, RectBivariateSpline
from scipy.misc import imrotate
from scipy.integrate import trapz
from astropy.io import fits

global px_scale

def psf_to_prf(dat, dx, dy, N_pix):
    '''
    Integrate PSFs up to PRFs.

    In:

        dat: (np.array, N_pix*N_pix), 2min PSF stack.
        dx, dy: sub-pixel centroid offset. (1/10th to 9/10ths of a pixel)
        N_pix: total number of pixels in PRF image

    Out:
        frac_1d: normalized 1*N_pix^2 array of PRF values.
    '''

    tot = trapz(trapz(dat))

    N_pix_per_side = int( (N_pix)**(1/2) )
    assert int( (N_pix)**(1/2) ) == float( (N_pix)**(1/2) )

    frac_2d = np.zeros((N_pix_per_side, N_pix_per_side))

    # The PRF is the integral of the PSF over each pixel.
    # Note that for pixels at the boundaries this is a crude indexing scheme,
    # but they will be zero anyway.
    for i in range(N_pix_per_side):
        for j in range(N_pix_per_side):

            this_x = np.array(range(i*px_scale, (i+1)*px_scale)) + \
                     np.round(dx * px_scale / 10).astype(int)
            this_y = np.array(range(j*px_scale, (j+1)*px_scale)) + \
                     np.round(dy * px_scale / 10).astype(int)

            frac_2d[i,j] = trapz(trapz(dat[this_x.min():this_x.max(),\
                                           this_y.min():this_y.max()]))

    frac_1d = frac_2d.reshape(1, N_pix_per_side**2)/tot

    return frac_1d


if __name__ == '__main__':

    #######################
    # MANUAL INPUT VALUES #
    #######################

    quick = False # boolean value for whether you want this code to run fast.
    field_angles = np.array([0, 6, 12, 17]) # Deb Woods, 17/05/12 email.
    psf_dir = '../data/PSFs_from_Deb_sent_May_17_2017/' # comments are below.
    psf_file = 'CamSN06_CCD1_16May2017_PSFnorm_bfd3p346focus_+0.03.mat'
    N_filters = 9 # number of wavelength filters.
    N_rotations = 2 # two different azimuthal orientations.
    px_scale = 101 # micron/px.
    qstr = '_quick' if quick else ''
    outname = 'bigfrac{:s}.fits'.format(qstr)

    #####################
    # END MANUAL INPUTS #
    #####################

    # Following Roland's science team meeting report (slides from the 28-29 Sep,
    # 2017 meeting), the current focus shift is ~20-30microns across all cameras,
    # and it appears to stabilize over time.

    # The convention for the sign is: negative defocus values mean the distance
    # between the detector and the last lens surface is increasing; positive
    # defocus values mean the distance between the detector and the last lens
    # surface is decreasing.

    # So we will use Deb Woods' -30 micron defocused PSFs. These were delivered
    # to me circa May 17, 2017. The sign convention in Deb's file names is OFC
    # opposite the true sign (per her May 9, 2017 email).
    psf_dir = psf_dir
    psf_file = psf_file
    N_field_angles = len(field_angles)

    # Number of wavelength filters. The struct `PSF_stellar` has the spectral
    # summing done for the following 9 types:
    #'F0 (7200)','F5 (6440)','G0 (6030)','G5 (5770)','K0 (5250)','K5 (4350)',
    #'M1 (3600)','M3 (3250)','M5 (2800)'
    N_filters = N_filters

    #####################
    # Read in the PSFs. #
    #####################
    # Matlab's `.mat` is a hdf5 variant, and everything gets saved as HDF5 object
    # references. We need to dereference them to read them into numpy arrays.
    f = h5py.File(psf_dir+psf_file, 'r')

    # f has keys: ['#refs#', 'PSF_README', 'PSF_lambda', 'PSF_stellar']
    # The only one used by Peter was `PSF_lambda` (really, PRF_lambda, but that was
    # a typo on Deb's part). Deb later changed this so that PSF_stellar has the
    # stellar-wavelength dependent PSF.

    #In [6]: list(f.get('PSF_lambda'))
    #Out[6]: ['PSF', 'field_angles', 'field_position', 'wavelength']

    #In [27]: f.get('PSF_stellar/PSF').shape
    #Out[27]: (9, 4)
    # the number of wavelength filters, by the number of field angles.

    image_size = f[f.get('PSF_stellar/PSF')[0,0]].shape
    image_len = image_size[0]

    N_pix = int((image_len / px_scale)**2) # number of TESS pixels in PRF image
    assert int(image_len/px_scale) == float(image_len/px_scale)

    x, y = np.array(range(image_len)), np.array(range(image_len))

    #####################################
    # Read in pointing error timeseries #
    #####################################
    df = pd.read_csv('../data/atterr.dat', delimiter=' ', header=None)
    df.columns = ['time','errx','erry']
    errx, erry = np.array(df['errx'])*px_scale, np.array(df['erry'])*px_scale

    # Time step in atterr.dat is 0.2 s, so need 600 points for 2-minute stack.
    N_jit_pts = 600

    # Errors are in pixels, so mult by px_scale for sub-pixel sampling. Also, take
    # the points from after the beginning of the timeseries (so that the behavior
    # converges to a "typical" section of flight).
    errx = errx[5000:5000+N_jit_pts]
    erry = erry[5000:5000+N_jit_pts]
    errx -= np.median(errx)
    erry -= np.median(erry)

    # `big_frac` is a numpy array of the PRF values, as a function of pixel index,
    # wavelength filter, field angle, and a randomly selected (x,y) pair that
    # spatially shifts the PRF.
    big_frac = np.zeros((20, 10, N_field_angles, N_pix, N_filters))

    for rot_i in range(N_rotations):
        for ang_i in range(N_field_angles):
            for filt_i in range(N_filters):

                if quick:
                    if ang_i > 0:
                        continue
                    if filt_i != 4:
                        continue

                print(
                        'rot: {:d}, angle: {:d}, filter: {:d}'.format(
                        rot_i, ang_i, filt_i)
                     )

                this_dat = np.array(f[f.get('PSF_stellar/PSF')[filt_i,ang_i]])

                func = interp2d(x, y, this_dat,
                        kind='linear', bounds_error=False, fill_value=0)

                shift_dat = np.zeros_like(this_dat)

                # stack the PSFs over the 2 minute exposure (w/ jitter).
                for jit_i in range(N_jit_pts):

                    if jit_i % 100 == 0:
                        print('\tjitter {:d}'.format(jit_i))

                    shift_dat += func( x+errx[jit_i], y+erry[jit_i] )

                # rotate the entire stack, if appropriate
                if (rot_i == 1) or (ang_i == 3):

                    # no "crop" of rotated image needed
                    assert shift_dat.shape[0] == shift_dat.shape[1]

                    shift_dat = imrotate(shift_dat, 45., 'bilinear')

                # subpixel centroid offsetting
                for dx in range(10):
                    for dy in range(10):

                        frac = psf_to_prf(shift_dat, dx, dy, N_pix)

                        big_frac[dx+rot_i*10, dy, ang_i, :, filt_i] = frac


    hdu = fits.PrimaryHDU()
    hdu.data = big_frac
    hdu.writeto('../results/{:s}'.format(outname), overwrite=True)
