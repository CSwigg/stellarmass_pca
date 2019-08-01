
import read_results as rr
import matplotlib.pyplot as plt
import numpy
from astropy.io import fits
r = rr.PCAOutput().from_fname('9894-9102_res.fits')
q = rr.qtyFig()

ax1 = plt.axes()


# m, s, mcb, scb, scale =  q.qtyFig(r, 'MLi', plt.axes(), plt.axes())

