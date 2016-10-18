import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import trapz

from astropy import units as u, constants as c, table as t
import astropy.io


l_eff_d = {'r': 6166. * u.AA, 'i': 7480. * u.AA, 'z': 8932. * u.AA}
l_wid_d = {'r': 550. * u.AA, 'i': 1300. * u.AA, 'z': 1000. * u.AA}


def lumdens2bbdlum(lam, Llam, band):
    '''
    Convert a spectral luminosity density to a broadband luminosity

    Convolve source spectrum/a with a filter; specific routine derived
        from MaNGA's ml_mangatosdssimage.pro routine from DRP

    Parameters
    ----------
    lam :
        wavelength array of source spectrum

    Llam :
        Luminosity density, units sim to Lsun/AA

    band :
        chosen band
    '''

    # make sure everything's in the right units
    Llam = Llam.to('erg s-1 AA-1')
    lam = lam.to('AA')
    nu = lam.to('Hz', equivalencies=u.spectral())

    # read in filter table
    band_tab = t.Table.read('filters/{}_SDSS.res'.format(band),
                            names=['lam', 'f'], format='ascii')

    # set up interpolator
    band_interp = interp1d(x=band_tab['lam'].quantity.value,
                           y=band_tab['f'], fill_value=0.,
                           bounds_error=False)

    f = band_interp(lam)

    # convert to Lnu, by multiplying by lam^2/c
    Lnu = (Llam * lam**2. / c.c).to('Lsun Hz-1')

    L = trapz(x=nu.value[::-1], y=f * Lnu.value, axis=-1) * u.Lsun
    Lnu_avg = L / (trapz(x=nu.value[::-1], y=f) * u.Hz)

    l_eff = l_eff_d[band]
    L = Lnu_avg * l_eff.to('Hz', equivalencies=u.spectral())

    return L.to('Lsun')

def color(hdulist, band1='g', band2='r'):
    '''
    Calculate the color of a MaNGA galaxy, based on two bandpasses

    By convention, C_br = b - r, i.e., band1 - band2
    '''
    img1 = hdulist['{}IMG'.format(band1)].data
    img2 = hdulist['{}IMG'.format(band2)].data

    color = -2.5 * np.log10(img1 / img2)

    return color

color_ml_conv = '''
 C     a_g   b_g    a_r   b_r    a_i   b_i    a_z   b_z    a_J   b_J    a_H   b_H    a_K   b_K
'ug'  -.221  .485  -.099  .345  -.053  .268  -.105  .226  -.128  .169  -.209  .133  -.260  .123
'ur'  -.390  .417  -.223  .299  -.151  .233  -.178  .192  -.172  .138  -.237  .104  -.273  .091
'ui'  -.375  .359  -.212  .257  -.144  .201  -.171  .165  -.169  .119  -.233  .090  -.267  .077
'uz'  -.400  .332  -.232  .239  -.161  .187  -.179  .151  -.163  .105  -.205  .071  -.232  .056
'gr'  -.499  1.519 -.306 1.097  -.222 0.864  -.223  .689  -.172  .444  -.189  .266  -.209  .197
'gi'  -.379  .914  -.220  .661  -.152  .518  -.175  .421  -.153  .283  -.186  .179  -.211  .137
'gz'  -.367  .698  -.215  .508  -.153  .402  -.171  .322  -.097  .175  -.117  .083  -.138  .047
'ri'  -.106  1.982 -.022  1.431  .006  1.114 -.052  .923  -.079  .650  -.148  .437  -.186  .349
'rz'  -.124  1.067 -.041  .780  -.018  .623  -.041  .463  -.011  .224  -.059  .076  -.092  .019
'BV'  -.942  1.737 -.628  1.305 -.520  1.094 -.399  .824  -.261  .433  -.209  .210  -.206  .135
'BR'  -.976  1.111 -.633  .816  -.523  .683  -.405  .518  -.289  .297  -.262  .180  -.264  .138
'''

C_ML_conv_t = astropy.io.ascii.read(color_ml_conv, guess=True, quotechar="'")
C_ML_conv_t.add_index('C')
