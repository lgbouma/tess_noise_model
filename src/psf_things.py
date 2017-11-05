'''
This code opens the product of the dithering routine written by Peter, which
takes Deb's PSFs, jitters and stacks them.

NAXIS1  =                   10 / length of data axis 1
NAXIS2  =                   20 / length of data axis 2
NAXIS3  =                    4 / length of data axis 3
NAXIS4  =                  144 / length of data axis 4
NAXIS5  =                    9 / length of data axis 5

PRF file (for us, as-built from deb woods). [10, 20, 4, 144, 9]:

4 field angles.
9 wavelengths (says S15 Fig12).
12 Teffs (Says S15 table 1).
144 total pixels in image.

10 x 20: some kind of x and y index. Each star gets this as a random number
between 1-10, and 1-20.

The "10x20" part might be what Peter refers to as the "sub-pixel centroid
offset, being stacked over different orientations":
"We do so over a 10x10 grid of sub-pixel centroid offsets and two different
azimuthal orientations (0° and 45°) with respect to the pixel boundaries."

The code never actually does any stacking though from what I can tell...

for i over stars:
    prf_this = reform(prf[dx[i],dy[i],fov_ind[i],*,0:(nfilt-1)])
'''
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import colors as colors

hdulist = fits.open('../data/dfrac_asbuilt_75c_0f.fits')
print(repr(hdulist[0].header))

data = np.array(hdulist[0].data)
#In [22]: data.shape
#Out[22]: (9, 144, 4, 20, 10)

X, Y = np.mgrid[0:12, 0:12]

for field_angle_index in np.arange(0,4,1):

    for x_centroid_jitter in np.arange(0,20,1):
        for y_centroid_jitter in np.arange(0,10,1):

            fai = field_angle_index
            xcj = x_centroid_jitter
            ycj = y_centroid_jitter

            print(fai, xcj, ycj)

            plt.close('all')

            n_wvlen = 9
            img = (np.sum(data[:,:,fai,xcj,ycj],axis=0)/n_wvlen).reshape(12,12)

            f,ax = plt.subplots(figsize=(4.8,4))

            ais = ax.pcolor(X, Y, img, cmap='gray')

            f.colorbar(ais, ax=ax, label='prf value (sum over $\lambda$)')

            savdir = '../results/psf_tests/'
            savname = 'field_ind_{:d}/xcen_{:d}_ycen{:d}.png'.format(
                    int(fai), int(xcj), int(ycj))

            f.savefig(savname, bbox_inches='tight')
