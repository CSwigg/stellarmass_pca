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

import figures_tools
import matplotlib.pyplot as plt
from matplotlib import gridspec

from astropy import coordinates as coord
from astropy import wcs
import astropy.units as u

import yaml


with open('config.yaml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)
    TeX = cfg['TeX']

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
        fits : fits.HDUList
            A constructed FITS HDUList Object
        fname : str
            Name of FITS file
        
        Returns
        -------
        ret : __main__.PCAOutput
            Returns construction of PCAOutput() instance
        """
        ret = super().fromfile(fname, *args, **kwargs)
        return ret

    @classmethod
    def from_plateifu(cls, basedir, plate, ifu, *args, **kwargs):
        """ Constructs an instanceof PCAOutput form a given results FITS file. 
        
        Parameters
        ----------
        basedir : str
            Directory to base of all plates
        plate : str
            Number of plate
        ifu : [type]
            Number of ifu
        
        Returns
        -------
        ret : __main__.PCAOutput
            Returns construction of PCAOutput() instance
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


    def quantity_figure(self, extname):
        
        return qtyFig().make_qty_fig(self, extname, fits.open('/Users/admin/Desktop/stellarmass_pca/manga-8144-3704-MAPS-HYB10-GAU-MILESHC.fits.gz'))
    
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

    @property
    def mask(self):
        """Creates a mask for a given map based on the map's success parameter for each spaxel.
        
        Returns
        -------
        numpy.ndarray
            Numpy array of boolean values representing a data mask
        """
        return np.logical_or(self.getdata('mask').astype(bool),
                             ~self.getdata('success').astype(bool))

    def badPDF(self, ch=2, thresh=1.0e-4):
        return self.cubechannel('GOODFRAC', ch) < thresh

    # Leave alone
    def get_drp_logcube(self, mpl_v):
        plateifu = self[0].header['PLATEIFU']
        drp = m.load_drp_logcube(*plateifu.split('-'), mpl_v)
        return drp
    # Leave alone
    def get_dap_maps(self, mpl_v, kind):
        plateifu = self[0].header['PLATEIFU']
        drp = m.load_dap_maps(*plateifu.split('-'), mpl_v, kind)
        return drp


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

    """Reconstruct the stellar mass from a PCA results objec
    
    Parameters
    ----------
    res : [type]
        PCAOutput instance
    pca_system : [type]
        PCASystem instance
    cosmo : [type]
        instance of astropy.cosmology.Cosmology or subclass
    band : [type]
        'r', 'i', or 'z'
    z : int
        redshift
    
    Returns
    -------
    int
        Mass in units of solar masses for a given band 
    """
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
    

    def make_qty_fig(self, pca_out:PCAOutput, qty_str, dap = None, qty_tex=None, qty_fname=None, f=None,
                        logify=False, TeX_over=None):
        '''
        make a with a map of the quantity of interest

        params:
            - qty_str: string designating which quantity from self.metadata
            to access
            - qty_tex: valid TeX for plot
            - qty_fname: override for final filename (usually used when `f` is)
            - f: factor by which to multiply map
        '''
        if qty_fname is None:
            qty_fname = qty_str
    
        # if qty_tex is None:
        #     qty_tex = self.pca.metadata[qty_str].meta.get(
        #         'TeX', qty_str)

        if qty_tex is None:
            try:
                self.qty_tex = cfg['TeX'][qty_str]
            except:
                self.qty_tex = qty_str


        self.pca_out = pca_out
        try:   
            self.dap = dap
            # dap[3] is just a convenient choice since it only has 2 dimensions
            self.wcs_header = wcs.WCS(dap[3])
        except:
            # TODO: handle conditions where no DAP product is passed in 
            pass
        self.map_shape = self.pca_out.getdata('Mask').shape
        self.mask = self.pca_out.mask
        fig, gs, ax1, ax2 = self.__setup_qty_fig__()
        
        m, s, mcb, scb, scale = self.qty_map(
            qty_str=qty_str, ax1=ax1, ax2=ax2, f=f, logify=logify,
            TeX_over=TeX_over)

        # fig.suptitle('{}: {}'.format(self.objname, qty_tex))

        self.__fix_im_axs__([ax1, ax2])
        

        #TODO: find self.objname
        # fname = '{}-{}.png'.format('self.objname', qty_fname)
        fname = '{}-{}.png'.format('test_fig', qty_fname)
        # TODO: find self.figdir
        # plt.savefig(fig, fname, self.figdir, dpi=300)

        plt.savefig('/Users/admin/Desktop/stellarmass_pca/' + fname, overwrite=True, dpi=300)
        return fig

    def qty_map(self, qty_str, ax1, ax2, f=None, norm=[None, None],
                logify=False, TeX_over=None):
       
        # plt.style.use('figures.mplstyle')
        '''
        make a map of the quantity of interest, based on the constructed
            parameter PDF

        params:
         - qty_str: string designating which quantity from self.metadata
            to access
         - ax1: where median map gets shown
         - ax2: where sigma map gets shown
         - f: factor to multiply percentiles by
         - log: whether to take log10 of
        '''
        

        d = self.pca_out[qty_str].data
        P50, l_unc, u_unc = d[0], d[1], d[2]
        
        scale = 'log'
        # P50, l_unc, u_unc, scale = self.pca.param_cred_intvl(
        #     qty=qty_str, factor=f, W=self.w,
        #     mask=np.logical_or(self.mask_map, ~self.fit_success))

        # if not TeX_over:
        #     med_TeX = self.pca.metadata[qty_str].meta.get('TeX', qty_str)
        # else:
        #     med_TeX = TeX_over
        med_TeX = self.qty_tex
        # manage logs for computation and display simultaneously
        if logify and (scale == 'log'):
            raise ValueError('don\'t double-log a quantity!')
        elif logify:
            P50 = np.log10(P50)
            unc = np.log10((u_unc + l_unc) / 2.)
            sigma_TeX = r'$\sigma~{\rm [dex]}$'
            #med_TeX = ''.join((r'$\log$', med_TeX))
        elif (scale == 'log'):
            unc = (u_unc + l_unc) / 2.
            sigma_TeX = r'$\sigma~{\rm [dex]}$'
            #med_TeX = ''.join((r'$\log$', med_TeX))
        else:
            unc = (l_unc + u_unc) / 2.
            sigma_TeX = r'$\sigma$'

        m_vmin, m_vmax = np.percentile(np.ma.array(P50, mask=self.mask).compressed(), [2., 98.])
        m = ax1.imshow(
            np.ma.array(P50, mask=self.mask),
            aspect='equal', norm=norm[0], vmin=m_vmin, vmax=m_vmax)

        s_vmin, s_vmax = np.percentile(np.ma.array(unc, mask=self.mask).compressed(),
                                       [2., 98.])
        s = ax2.imshow(
            np.ma.array(unc, mask=self.mask),
            aspect='equal', norm=norm[1], vmin=s_vmin, vmax=s_vmax)

        mcb = plt.colorbar(m, ax=ax1, fraction=0.046, pad=0.04)
        mcb.set_label(med_TeX, size='xx-small')
        mcb.ax.tick_params(labelsize='xx-small')

        scb = plt.colorbar(s, ax=ax2, fraction=0.046, pad=0.04)
        scb.set_label(sigma_TeX, size=8)
        scb.ax.tick_params(labelsize='xx-small')

        return m, s, mcb, scb, scale


    def __fix_im_axs__(self, axs, bad=True):
        '''
        do all the fixes to make quantity maps look nice in wcsaxes
        '''
        if type(axs) is not list:
            axs = [axs]

        # create a sky offset frame to overlay
        offset_frame = coord.SkyOffsetFrame(
            origin=coord.SkyCoord(*(self.wcs_header.wcs.crval * u.deg)))

        # over ax objects
        for ax in axs:
            # suppress native coordinate system ticks
            for ci in range(2):
                ax.coords[ci].set_ticks(number=5)
                ax.coords[ci].set_ticks_visible(False)
                ax.coords[ci].set_ticklabel_visible(False)
                ax.coords[ci].grid(False)

            # initialize overlay
            offset_overlay = ax.get_coords_overlay(offset_frame)
            offset_overlay.grid(True)
            offset_overlay['lon'].set_coord_type('longitude', coord_wrap=180.)
            ax.set_aspect('equal')

            for ck, abbr, pos in zip(['lon', 'lat'], [r'\alpha', r'\delta'], ['b', 'l']):
                offset_overlay[ck].set_axislabel(
                    r'$\Delta {}~["]$'.format(abbr), size='x-small')
                offset_overlay[ck].set_axislabel_position(pos)
                offset_overlay[ck].set_ticks_position(pos)
                offset_overlay[ck].set_ticklabel_position(pos)
                offset_overlay[ck].set_format_unit(u.arcsec)
                offset_overlay[ck].set_ticks(number=5)
                offset_overlay[ck].set_major_formatter('s.s')

            if bad:
                # figures_tools.annotate_badPDF(ax, self.goodPDF)
                pass
    def __setup_qty_fig__(self):
        fig = plt.figure(figsize=(8, 6), dpi=80)
        gs = gridspec.GridSpec(1, 2, wspace=.4, left=.12, right=.93,
                               bottom=.10, top=.85)
        ax1 = fig.add_subplot(gs[0], projection=self.wcs_header)
        ax2 = fig.add_subplot(gs[1], projection=self.wcs_header)
        
        # overplot hatches for masks
        # start by defining I & J pixel grid
        II, JJ = np.meshgrid(*(np.linspace(-.5, ms_ - .5, ms_ + 1)
                               for ms_ in self.map_shape))
        IIc, JJc = map(lambda x: 0.5 * (x[:-1, :-1] + x[1:, 1:]), (II, JJ))
       
        for ax in [ax1, ax2]:
            # pcolor plots are masked where the data are GOOD
            # badpdf mask
            ax.pcolor(II, JJ,
                      np.ma.array(np.zeros_like(IIc), mask=~self.pca_out.badPDF()),
                      hatch='\\'*8, alpha=0.)
            # dered mask
            ax.pcolor(II, JJ,
                      np.ma.array(np.zeros_like(IIc), mask=~self.mask),
                      hatch='/'*8, alpha=0.)
            # fit unsuccessful
            ax.pcolor(II, JJ,
                      np.ma.array(np.zeros_like(IIc), mask=self.pca_out['SUCCESS']),
                      hatch='.' * 8, alpha=0.)

        return fig, gs, ax1, ax2

result = PCAOutput.from_fname('/Users/admin/Desktop/stellarmass_pca/8144-3704_res.fits')
result.quantity_figure('MLi')