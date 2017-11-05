'''
plots the TESS spacecraft pointing error, from a data file Peter received from
someone at Orbital some time in 2015.
'''
import pandas as pd, numpy as np, matplotlib.pyplot as plt

f, ax = plt.subplots()

df = pd.read_csv('../data/atterr.dat', delimiter=' ', header=None)
df.columns = ['time','errx','erry']
errx, erry = np.array(df['errx']), np.array(df['erry'])

ax.plot(errx)
ax.plot(erry+0.2)

ax.set_xlabel('N time steps (each is 0.2sec)')
ax.set_ylabel('pointing err (pixels)')

f.savefig('../results/spacecraft_pointing_error.pdf')
