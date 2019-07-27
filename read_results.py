import numpy as np
from scipy.stats import skewnorm
import scipy.optimize as spopt

from astropy.io import fits
from astropy import nddata, wcs

from importer import *
import utils as ut
import spectrophot

import manga_tools as m

from glob import glob
import os
from functools import lru_cache

class PCASystem(fits.HDUList):
    @property
    def M(self):
        return self['MEAN'].data

    @property
    def E(self):
        return self['EVECS'].data

    @property
    def l(self):
        return self['LAM'].data

    @property
    def logl(self):
        return np.log10(self.l)

    @property
    def dlogl(self):
        return ut.determine_dlogl(self.logl)

class PCAOutput(fits.HDUList):
    '''
    stores output data from PCA that as been written to FITS.
    '''
    @classmethod
    def from_fname(cls, fname, *args, **kwargs):
        """ Constructs an instance of PCAOutput from a given results FITS file.
        
        Parameters
        ----------
        fits : [type]
            [description]
        fname : str
            Name of FITS file
        
        Returns
        -------
        ret : __main__.PCAOutput
            Returned construction of PCAOutput() instance
        """
        ret = super().fromfile(fname, *args, **kwargs)
        return ret

    @classmethod
    def from_plateifu(cls, basedir, plate, ifu, *args, **kwargs):
        """ Constructs an instanceof PCAOutput form a given results FITS file. 
        
        Parameters
        ----------
        basedir : [type]
            [description]
        plate : str
            [description]
        ifu : [type]
            [description]
        
        Returns
        -------
        [type]
            [description]
        """
        fname = os.path.join(basedir, '{}-{}'.format(plate, ifu),
                             '{}-{}_res.fits'.format(plate, ifu))
        return super().fromfile(fname, *args, **kwargs)

    def getdata(self, extname):
        """ Accesses array of data values of PCAOutput instance with given extension.
        
        Parameters
        ----------
        extname : str
            HDU extension name
        
        Returns
        -------
        numpy.ndarray
            NumPy array of accessed data
        """

        return self[extname].data

    def flattenedmap(self, extname):
        """ Flattenes 2D array of data from HDU extension into 1D array. 
        
        Parameters
        ----------
        extname : str
            HDU extension name
        
        Returns
        -------
        numpy.ndarray
            Flattened NumPy array of accessed data
        """
        return self.getdata(extname).flatten()

    def cubechannel(self, extname, ch):
        """ Get one channel (indexed along axis 0) of one extension
        
        Parameters
        ----------
        extname : str
            HDU extension name
        ch : int
            Extension channel index 
        
        Returns
        -------
        numpy.ndarray
            NumPy array of data at specified HDU extension channel
        """
        
        return self.getdata(extname)[ch]

    def flattenedcubechannel(self, extname, ch):
        """ Flattenes 2D array of data from HDU extension channel into 1D array.
        
        Parameters
        ----------
        extname : str
            HDU extension name
        ch : int
            Extension channel index
        
        Returns
        -------
        numpy.ndarray
            Flattened NumPy array of accessed data
        """
        return self.cubechannel(extname, ch).flatten()

    def flattenedcubechannels(self, extname, chs):
        return np.stack([self.flattenedcubechannel(extname, ch)
                         for ch in chs])

    def param_dist_med(self, extname, flatten=False):
        """Accesses the median (Oth) channel of an extension
        
        Parameters
        ----------
        extname : str
            HDU extension name 
        flatten : bool, optional
            Flattens 2D array into 1D, by default False
        
        Returns
        -------
        numpy.ndarray
            Median (0th) channel of specified HDU extension
        """
        med = self.cubechannel(extname, 0)
        if flatten:
            med = med.flatten()

        return med

    def param_dist_wid(self, extname, flatten=False):
        """[summary]
        
        Parameters
        ----------
        extname : [type]
            [description]
        flatten : bool, optional
            [description], by default False
        
        Returns
        -------
        [type]
            [description]
        """
        distwid = self.cubechannel(extname, 2) + self.cubechannel(extname, 1)
        if flatten:
            distwid = distwid.flatten()

        return distwid

    def setup_photometry(self, pca_system):
        """
        
        Parameters
        ----------
        pca_system : PCASystem() 
            Instance of PCASystem
        """
        fitcube_f = self.getdata('NORM') * \
                    (np.einsum('al,a...->l...', pca_system.E, self.getdata('CALPHA')) + \
                     pca_system.M[:, None, None]) * m.spec_unit
        self.spec2phot = spectrophot.Spec2Phot(
            lam=pca_system.l * m.l_unit, flam=fitcube_f, axis=0)

    def get_color(self, b1, b2, filterset='sdss2010', flatten=False):
        if not hasattr(self, 'spec2phot'):
            raise AttributeError('no spec2phot object initialized')

        b1 = '-'.join((filterset, b1))
        b2 = '-'.join((filterset, b2))

        color = self.spec2phot.color(b1, b2)

        if flatten:
            color = color.flatten()

        return color

    @property
    def mask(self):
        return np.logical_or(self.getdata('mask').astype(bool),
                             ~self.getdata('success').astype(bool))

    def badPDF(self, ch=2, thresh=1.0e-4):
        return self.cubechannel('GOODFRAC', ch) < thresh

    def get_drp_logcube(self, mpl_v):
        plateifu = self[0].header['PLATEIFU']
        drp = m.load_drp_logcube(*plateifu.split('-'), mpl_v)
        return drp

    def get_dap_maps(self, mpl_v, kind):
        plateifu = self[0].header['PLATEIFU']
        drp = m.load_dap_maps(*plateifu.split('-'), mpl_v, kind)
        return drp

    def to_normaldist(self, extname):
        mu = self.param_dist_med(extname)
        sd = 0.5 * self.param_dist_wid(extname)
        return mu, sd


class MocksPCAOutput(PCAOutput):
    '''
    PCA output for mocks
    '''

    def truth(self, extname, flatten=False):
        truth = self[extname].header['TRUTH']
        ret = truth * np.ones_like(self['SNRMED'].data)
        if flatten:
            ret = ret.flatten()
        return ret

    def dev_from_truth(self, extname, flatten=False):
        truth = self[extname].header['TRUTH']
        return self.param_dist_med(extname, flatten) - truth

    def dev_from_truth_div_distwid(self, extname, flatten=False):
        distwid = self.param_dist_wid(extname, flatten)
        dev = self.dev_from_truth(extname, flatten)
        dev_distwid = dev / (0.5 * distwid)
        return dev / (0.5 * distwid)

def bandpass_mass(res, pca_system, cosmo, band, z):
    '''
    reconstruct the stellar mass from a PCA results object

    parameters:
     - res: PCAOutput instance
     - pca_system: PCASystem instance
     - cosmo: instance of astropy.cosmology.Cosmology or subclass
     - band: 'r', 'i', or 'z'
    '''
    spec2phot = fit_spec2phot(res, pca_system)

    # apparent magnitude
    band_mag = spec2phot.ABmags['sdss2010-{}'.format(band)]
    distmod = cosmo.distmod(z)

    # absolute magnitude
    band_MAG = band_mag - distmod.value
    band_sollum = 10.**(-0.4 * (band_MAG - spectrophot.absmag_sun_band[band])) * \
                  m.bandpass_sol_l_unit

    # mass-to-light (in bandpass solar units)
    masstolight = (10.**res.cubechannel('ML{}'.format(band), ch=0)) * m.m_to_l_unit

    mass = (masstolight * band_sollum).to('Msun')

    return mass

def fit_spec2phot(res, pca_system):
    '''
    set up a spectrum-to-photometric conversion for a given pca cube
    '''
        # construct best-fit cube
    fitcube_f = res.getdata('NORM') * \
                (np.einsum('al,a...->l...', pca_system.E, res.getdata('CALPHA')) + \
                 pca_system.M[:, None, None]) * m.spec_unit
    spec2phot = spectrophot.Spec2Phot(lam=pca_system.l * m.l_unit, flam=fitcube_f, axis=0)

    return spec2phot

class qtyFig():
    

    def __init__(self, res):

        self.res = res

    def make_plot(self, qty_str, qty_tex=None, qty_fname=None, f=None, logify=False, TeX_over=None):
        if qty_fname is None:
            qty_fname = qty_str

        if qty_tex is None:
            qty_tex = self.pca.metadata[qty_str].meta.get(
                'TeX', qty_str)