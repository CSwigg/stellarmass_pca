import numpy as np
import manga_tools as m

from astropy import wcs, nddata
from astropy import units as u, constants as c

def get_emline_qty(maps, qty, key, sn_th=3., maskbits=list(range(64))):
    '''
    return a masked array for a particular emission-line
    '''

    qty = qty.upper()
    qty_extname = 'EMLINE_{}'.format(qty)
    qty_ivar_extname = 'EMLINE_{}_IVAR'.format(qty)
    qty_mask_extname = 'EMLINE_{}_MASK'.format(qty)

    qty_hdu = maps[qty_extname]
    qty_ivar_hdu = maps[qty_ivar_extname]
    qty_mask_hdu = maps[qty_mask_extname]

    # get a mapping from eline key to channel key
    v2k = {v: k for (k, v) in qty_hdu.header.items()}
    # get a mapping from channel key to channel
    cstring2ix = lambda s: int(s[1:]) - 1

    ix = cstring2ix(v2k[key])

    meas = qty_hdu.data[ix, ...]
    ivar = qty_ivar_hdu.data[ix, ...]
    snr = meas * np.sqrt(ivar)
    snr_mask = (snr < sn_th)

    map_mask = m.mask_from_maskbits(qty_mask_hdu.data[ix, ...])
    mask = np.logical_or.reduce((snr_mask, map_mask))

    return np.ma.array(meas, mask=mask)

class MaNGAElines(object):
    units = {'GFLUX': '1e-17 erg s-1 cm-2',
             'SFLUX': '1e-17 erg s-1 cm-2',
             'GEW': 'AA', 'SEW': 'AA'}
    def __init__(self, hdulist, *args, **kwargs):
        self.hdulist = hdulist
        self.wcs = wcs.WCS(hdulist['BIN_AREA'].header)

    def get_qty(self, qty, key, sn_th, maskbits=[30]):
        maps = self.hdulist
        qty = qty.upper()
        qty_extname = 'EMLINE_{}'.format(qty)
        qty_ivar_extname = 'EMLINE_{}_IVAR'.format(qty)
        qty_mask_extname = 'EMLINE_{}_MASK'.format(qty)

        qty_hdu = maps[qty_extname]
        qty_ivar_hdu = maps[qty_ivar_extname]
        qty_mask_hdu = maps[qty_mask_extname]

        # get a mapping from eline key to channel key
        v2k = {v: k for (k, v) in qty_hdu.header.items()}
        # get a mapping from channel key to channel
        cstring2ix = lambda s: int(s[1:]) - 1

        ix = cstring2ix(v2k[key])

        meas = qty_hdu.data[ix, ...]
        ivar = qty_ivar_hdu.data[ix, ...]
        ivar_zero = (ivar == 0.)
        std = 1. / np.sqrt(ivar.clip(min=1.0e-20))
        std[ivar_zero] = 0.

        snr_mask = ((meas * np.sqrt(ivar)) < sn_th)

        map_mask = m.mask_from_maskbits(qty_mask_hdu.data[ix, ...])
        mask = np.logical_or.reduce((ivar_zero, map_mask, snr_mask))

        unit = self.units[qty]

        data = nddata.NDDataRef(
            meas, uncertainty=nddata.StdDevUncertainty(std, unit=unit),
            unit=unit, mask=mask, wcs=wcs)

        return data

    @classmethod
    def DAP_from_plateifu(cls, plate, ifu, mpl_v, kind, *args, **kwargs):
        hdulist = m.load_dap_maps(plate, ifu, mpl_v, kind)
        return cls(hdulist, *args, **kwargs)
