
import read_results as rr
import matplotlib.pyplot as plt
import numpy
from astropy.io import fits

r = rr.PCAOutput().from_fname('/Users/admin/Desktop/stellarmass_pca/8144-3704_res.fits')

q = rr.qtyFig()
m, s, mcb, scb, scale = q.qty_map(r, 'MLi', plt.axes(), plt.axes())
# Right now I'm only accessing qty_map, not make_qty_map yet.
plt.show(m)




