import numpy as np
from scipy.special import expit
from scipy.optimize import curve_fit

from warnings import warn, filterwarnings, catch_warnings, simplefilter
from functools import lru_cache

# plotting
import matplotlib.pyplot as plt
from matplotlib import cm as mplcm
from matplotlib import gridspec
import matplotlib.ticker as mticker

# astropy ecosystem
from astropy import constants as c, units as u, table as t
from astropy.io import fits
from astropy import wcs
from astropy import coordinates as coord
from astropy.wcs.utils import pixel_to_skycoord

import os
import sys

# sklearn
from sklearn.neighbors import KNeighborsRegressor

# local
from importer import *
import read_results
import spectrophot

# personal
import manga_tools as m
import spec_tools

spec_unit = 1e-17 * u.erg / u.s / u.cm**2. / u.AA
l_unit = u.AA
bandpass_sol_l_unit = u.def_unit(
    s='bandpass_solLum', format={'latex': r'\overbar{\mathcal{L}_{\odot}}'},
    prefixes=False)
m_to_l_unit = 1. * u.Msun / bandpass_sol_l_unit

band_ix = dict(zip('FNugriz', range(len('FNugriz'))))

sdss_bands = 'ugriz'
nsa_bands = 'FNugriz'

class Sigmoid(object):
    p0 = [70., .1, -5., 20.]

    def __init__(self, vscale, hscale, xoffset, yoffset):
        self.vscale, self.hscale = vscale, hscale
        self.xoffset, self.yoffset = xoffset, yoffset

    @staticmethod
    def sigmoid(x, vscale, hscale, xoffset, yoffset):
        return vscale * expit(x / hscale + xoffset) + yoffset

    @classmethod
    def from_points(cls, x, y):
        params, *_ = curve_fit(f=Sigmoid.sigmoid, xdata=x, ydata=y, 
                                p0=Sigmoid.p0)
        return cls(*params)

    def __call__(self, x):
        return self.sigmoid(x, self.vscale, self.hscale, self.xoffset, self.yoffset)

    def __repr__(self):
        return 'Sigmoid function: \
                <vscale = {:.02e}, hscale = {:.02e}, xoffset = {:.02e}, yoffset = {:.02e}>'.format(
                    self.vscale, self.hscale, self.xoffset, self.yoffset)

ba_to_majaxis_angle = Sigmoid.from_points(
    [-1000., 0.2, 0.45, 0.65, 0.9, 1000.], [20., 30., 50., 70., 90., 100.])

def infer_masked(q_trn, bad_trn, infer_here):
    '''
    infer the masked values in the interior of an IFU (foreground stars, dropped fibers)
    '''

    q_final = 1. * q_trn

    # coordinate arrays
    II, JJ = np.meshgrid(*list(map(np.arange, q_trn.shape)), indexing='ij')

    # set up KNN regressor
    knn = knn_regr(trn_coords=[II, JJ], trn_vals=q_trn, good_trn=~bad_trn)
    q_final[infer_here] = infer_from_knn(knn=knn, coords=(II, JJ))[infer_here]

    return q_final

def ifu_bandmag(s2p, b, low_or_no_cov, drp3dmask_interior):
    '''
    flux in bandpass, inferring at masked spaxels
    '''
    mag_band = s2p.ABmags['sdss2010-{}'.format(b)]
    nolight = ~np.isfinite(mag_band)
    # "interpolate" over spaxels with foreground stars or other non-inference-related masks
    mag_band = infer_masked(
        q_trn=mag_band, bad_trn=np.logical_or.reduce((low_or_no_cov, drp3dmask_interior, nolight)),
        infer_here=drp3dmask_interior)
    mag_band[nolight] = np.inf
    return mag_band

def bandpass_flux_to_solarunits(sun_flux_band):
    '''
    equivalency for bandpass fluxes and solar units
    '''
    sun_flux_band_Mgy = sun_flux_band.to(m.Mgy).value
    def convert_flux_to_solar(f):
        s = f / sun_flux_band_Mgy
        return s

    def convert_solar_to_flux(s):
        f = s * sun_flux_band_Mgy
        return f

    return [(m.Mgy, m.bandpass_sol_l_unit, convert_flux_to_solar, convert_solar_to_flux)]

class StellarMass(object):
    '''
    calculating galaxy stellar mass with incomplete measurements
    '''
    bands = 'griz'
    bands_ixs = dict(zip(bands, range(len(bands))))
    absmag_sun = np.array([spectrophot.absmag_sun_band[b] for b in bands]) * u.ABmag

    def __init__(self, results, pca_system, drp, dap, drpall_row, cosmo, mlband='i'):
        self.results = results
        self.pca_system = pca_system
        self.drp = drp
        self.dap = dap
        self.drpall_row = drpall_row
        self.cosmo = cosmo
        self.mlband = mlband

        with catch_warnings():
            simplefilter('ignore')
            self.results.setup_photometry(pca_system)

        self.s2p = results.spec2phot

        # stellar mass to light ratio
        self.ml0 = results.cubechannel('ML{}'.format(mlband), 0)
        self.badpdf = results.cubechannel('GOODFRAC', 2) < 1.0e-4
        self.ml_mask = np.logical_or.reduce((self.results.mask, self.badpdf))

        # infer values in interior spaxels affected by one of the following:
        # bad PDF, foreground star, dead fiber
        self.low_or_no_cov = m.mask_from_maskbits(
            self.drp['MASK'].data, [0, 1]).mean(axis=0) > .3
        self.drp3dmask_interior = m.mask_from_maskbits(
            self.drp['MASK'].data, [2, 3]).mean(axis=0) > .3
        self.interior_mask = np.logical_or.reduce((self.badpdf, self.drp3dmask_interior))

        self.logml_final = infer_masked(
            q_trn=self.ml0, bad_trn=self.ml_mask,
            infer_here=self.interior_mask) * u.dex(m.m_to_l_unit)

    @property
    @lru_cache(maxsize=128)
    def distmod(self):
        return self.cosmo.distmod(self.drpall_row['nsa_zdist'])

    @property
    @lru_cache(maxsize=128)
    def nsa_absmags(self):
        absmags = np.array([nsa_absmag(self.drpall_row, band, kind='elpetro')
                            for band in self.bands]) * (u.ABmag - u.MagUnit(u.littleh**2))
        return absmags

    @property
    @lru_cache(maxsize=128)
    def nsa_absmags_cosmocorr(self):
        return self.nsa_absmags.to(u.ABmag, u.with_H0(self.cosmo.H0))

    @property
    @lru_cache(maxsize=128)
    def mag_bands(self):
        return np.array([ifu_bandmag(self.s2p, b, self.low_or_no_cov, self.drp3dmask_interior)
                         for b in self.bands]) * u.ABmag

    @property
    @lru_cache(maxsize=128)
    def flux_bands(self):
        return self.mag_bands.to(m.Mgy)

    @property
    @lru_cache(maxsize=128)
    def absmag_bands(self):
        return self.mag_bands - self.distmod

    @property
    @lru_cache(maxsize=128)
    def ifu_flux_bands(self):
        return self.flux_bands.sum(axis=(1, 2))

    @property
    @lru_cache(maxsize=128)
    def ifu_mag_bands(self):
        return self.ifu_flux_bands.to(u.ABmag)

    @property
    @lru_cache(maxsize=128)
    def sollum_bands(self):
        return self.absmag_bands.to(
            u.dex(m.bandpass_sol_l_unit),
            bandpass_flux_to_solarunits(self.absmag_sun[..., None, None]))

    @property
    @lru_cache(maxsize=128)
    def logml_fnuwt(self):
        return np.average(
            self.logml_final.value,
            weights=self.flux_bands[self.bands_ixs[self.mlband]].value) * self.logml_final.unit

    @property
    @lru_cache(maxsize=128)
    def mstar(self):
        return (self.sollum_bands + self.logml_final).to(u.Msun)

    @property
    @lru_cache(maxsize=128)
    def mstar_in_ifu(self):
        return self.mstar.sum(axis=(1, 2))

    def to_table(self):
        '''
        make table of stellar-mass results

        [plateifu, [flux_ifusummed_band1, flux_ifusummed_band2], ...,
         [flux_nsa_band1, flux_nsa_band2], ...,
         [dflux_(nsa-ifu)_band1, dflux_(nsa-ifu)_band2], ...,
         mass_in_ifu, logml_apercorr_ring, logml_apercorr_cmlr, logml_fluxwtd]
        '''

        tab = t.Table()
        tab['plateifu'] = [self.drpall_row['plateifu']]

        # tabulate mass in IFU
        tab['mass_in_ifu'] = self.mstar_in_ifu[None, ...]
        

        # calculate and store missing luminosities in solar units
        flux_outside_ifu = (self.nsa_absmags_cosmocorr + self.distmod).to(m.Mgy) - self.ifu_flux_bands
        mag_outside_ifu = flux_outside_ifu.to(u.ABmag)
        absmag_outside_ifu = mag_outside_ifu - self.distmod
        sollum_outside_ifu = absmag_outside_ifu.to(
            u.dex(m.bandpass_sol_l_unit), bandpass_flux_to_solarunits(self.absmag_sun))
        tab['missing_lum'] = sollum_outside_ifu[None, ...]
        tab['missing_lum'].meta['band'] = self.bands

        tab['ml_ring'] = self.ml_ring

        tab['ml_fluxwt'] = self.logml_fnuwt
        tab['ml_fluxwt'].meta['band'] = self.mlband
        
        tab['ifu_lum'] = (self.ifu_mag_bands - self.distmod).to(
            u.dex(m.bandpass_sol_l_unit), bandpass_flux_to_solarunits(self.absmag_sun))[None, ...]
        tab['ifu_lum'].meta['band'] = self.mlband

        return tab

    @property
    @lru_cache(maxsize=128)
    def ml_ring(self):
        '''
        "ring" aperture-correction
        '''
        phi = self.dap['SPX_ELLCOO'].data[2, ...]
        angle_from_majoraxis = np.minimum.reduce(
            (np.abs(phi), np.abs(180. - phi), np.abs(360. - phi)))
        # how close to major axis must a spaxel be in order to consider it?
        close_to_majaxis = (angle_from_majoraxis <= ba_to_majaxis_angle(
            self.drpall_row['nsa_elpetro_ba']))
        reff = np.ma.array(
            self.dap['SPX_ELLCOO'].data[1], mask=np.logical_or(self.ml_mask, ~close_to_majaxis))
        outer_ring = np.logical_and.reduce((
            (reff <= reff.max()), (reff >= reff.max() - .5)))
        outer_logml_ring = np.median(self.ml0[~self.ml_mask * outer_ring]) * self.logml_final.unit

        return outer_logml_ring

def knn_regr(trn_coords, trn_vals, good_trn, k=8):
    '''
    return a trained estimator for k-nearest-neighbors

    - trn_coords: list containing row-coordinate map and col-coordinate map
    - trn_vals: map of values used for training
    - good_trn: binary map (True signifies good data)
    '''
    II, JJ = trn_coords
    good = good_trn.flatten()
    coo = np.column_stack([II.flatten()[good], JJ.flatten()[good]])
    vals = trn_vals.flatten()[good]

    knn = KNeighborsRegressor(
        n_neighbors=k, weights='uniform', p=2)
    knn.fit(coo, vals)

    return knn

def infer_from_knn(knn, coords):
    '''
    use KNN regressor to infer values over a grid
    '''
    II, JJ = coords
    coo = np.column_stack([II.flatten(), JJ.flatten()])
    vals = knn.predict(coo).reshape(II.shape)

    return vals

def apply_cmlr(ifu_s2p, cb1, cb2, mlrb, cmlr_poly, f_tot, exterior_mask,
               fluxes_keys='FNugriz'):
    '''
    apply a color-mass-to-light relation
    '''
    f_tot_d = dict(zip(fluxes_keys, f_tot))

    # find missing flux in color-band 1
    fb1_ifu = (ifu_s2p.ABmags['sdss2010-{}'.format(cb1)] * u.ABmag)[~exterior_mask].to(m.Mgy).sum()
    dfb1 = f_tot_d[cb1] - fb1_ifu
    # find missing flux in color-band 2
    fb2_ifu = (ifu_s2p.ABmags['sdss2010-{}'.format(cb2)] * u.ABmag)[~exterior_mask].to(m.Mgy).sum()
    dfb2 = f_tot_d[cb2] - fb2_ifu

    # if there's no missing flux in one/both bands, then this method fails
    if (dfb1.value <= 0.) or (dfb2.value <= 0.):
        return -np.inf

    c = dfb1.to(u.ABmag) - dfb2.to(u.ABmag)

    logml = np.polyval(p=cmlr_poly, x=c.value)

    return logml

def nsa_mass(drpall_row, band, kind='elpetro'):
    # kind is elpetro or sersic
    mass = drpall_row['nsa_{}_mass'.format(kind)][band_ix[band]]
    return mass

def nsa_flux(drpall_row, band, kind='elpetro'):
    # kind is petro, elpetro, or sersic
    flux = drpall_row['nsa_{}_flux'.format(kind)][band_ix[band]]
    return flux

def nsa_absmag(drpall_row, band, kind='elpetro'):
    # kind is petro, elpetro, or sersic
    flux = drpall_row['nsa_{}_absmag'.format(kind)][band_ix[band]]
    return flux
