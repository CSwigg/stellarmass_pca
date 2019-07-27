'''
some tools for working with astronomical spectra
'''

import numpy as np

from astropy import units as u, constants as c


def Dn4000_index(l, s):
    '''
    compute D4000 index

    params:
     - l: wavelength array [Angstroms], length n
     - s: spectrum [flux or flux density units], length n
    '''

    blue = [3850., 3950.]
    red = [4000., 4100.]

    sblue = s[(l > blue[0]) * (l < blue[-1]), ...]
    lblue = l[(l > blue[0]) * (l < blue[-1]), ...]
    sred = s[(l > red[0]) * (l < red[-1]), ...]
    lred = l[(l > red[0]) * (l < red[-1]), ...]

    fblue = np.trapz(x=lblue, y=(lblue**2.)[..., None, None] * sblue, axis=0)
    fblue /= (blue[-1] - blue[0])
    fred = np.trapz(x=lred, y=(lred**2.)[..., None, None] * sred, axis=0)
    fred /= (red[-1] - red[0])

    return fred / fblue


def Hdelta_A_index(l, s):
    '''
    compute HdA index

    define index wrt blue continuum and red continuum

    params:
     - l: wavelength array [Angstroms], length n
     - s: spectrum [flux or flux density units], length n
    '''

    dl = determine_dl(l)

    blue = [4041.600, 4079.750]
    red = [4128.500, 4161.000]
    line = [4083.500, 4122.250]

    lred_m = (l > red[0]) * (l < red[-1])
    lblue_m = (l > blue[0]) * (l < blue[-1])
    lline_m = (l > line[0]) * (l < line[-1])

    lred = l[lred_m]
    lblue = l[lblue_m]
    lline = l[lline_m]
    sline = s[lline_m]

    # values at interpolation limits
    red_avg = s[lred_m, ...].mean(axis=0)
    blue_avg = s[lblue_m, ...].mean(axis=0)
    cont_slope = (red_avg - blue_avg) / (np.mean(red) - np.mean(blue))
    # expected continuum flux-density within line
    cont_interp = lambda l: ((cont_slope[None, ...] *
                             (l - np.mean(blue))[..., None, None]) +
                             blue_avg[None, ...])
    # use trapezoidal rule to approximate deficit in continuum flux-density
    EW = np.trapz(x=l[lline_m],
                  y=(1. - ((sline * (lline**2.)[..., None, None]) /
                           (cont_interp(lline) * (lline**2.)[..., None, None]))),
                  axis=0)

    return EW


def shift_to_rest_roll(v, dlogl=None):
    '''
    return number of pixels of roll needed to cancel out doppler shift
    '''
    if dlogl is None:
        dlogl = np.round(np.mean(logl[1:] - logl[:-1]), 8)

    z = (v / c.c).to('').value
    npix = -int(np.rint(z / dlogl))
    return npix


def shift_to_rest(logl, v):
    '''
    apply a constant velocity offset to a log-wavelength grid
    (i.e., DE-redshift the grid)
    '''
    z = (v / c.c).to('').value
    return logl - z


def determine_dlogl(logl):
    dlogl = np.round(np.mean(logl[1:] - logl[:-1]), 8)
    return dlogl


def determine_dl(l):
    logl = np.log10(l)
    dlogl = determine_dlogl(logl)
    llllim, llulim = logl - dlogl/2., logl + dlogl/2.
    lllim, lulim = np.log10(llllim), np.log10(llulim)
    dl = lulim - lllim
    return dl


def cube2rss(a, axis=0):
    '''
    transform cube-shaped data into RSS-shaped data
    '''
    if type(a) is not list:
        a = [a, ]

    a = np.row_stack([a_.reshape((-1, ) + (a_.shape[axis], )).T for a_ in a])
    return a

def air2vac(l, unit=None):
    '''
    calculate vacuum wavelengths, from air wavelengths

    based on airtovac (NASA IDLAstro)
    '''

    if unit is not None:
        l = l * u.Unit(unit)

    l = l.to('AA').value

    sigma2 = (1.0e4 / l)**2.
    n = 1. + .000065328 + .0294981 / (146. - sigma2) + (.0002554 / (41. - sigma2))
    vac_l = l * n

    # f = 1. + 5.792105e-2 / (238.0185 - sigma2) + 1.67917e-3 / (57.362 - sigma2)
    # vac_l = f * n

    return vac_l * u.AA

def vac2air(l):
    '''
    calculate air wavelengths, from vacuum wavelengths

    based on SDSS routine
    (http://classic.sdss.org/dr7/products/spectra/vacwavelength.html)
    '''

    l = l.to('AA').value

    air_l = l / (1.0 + 2.735182E-4 + 131.4182 / l**2. + 2.76249E8 / l**4.)

    return air_l * u.AA
