import os, sys, matplotlib
import faulthandler; faulthandler.enable()

mpl_v = 'MPL-8'
daptype = 'SPX-MILESHC-MILESHC'
os.environ['STELLARMASS_PCA_RESULTSDIR'] = '/Users/admin/sas/dr16/manga/spectro/mangapca/1.0.0/v2_5_3/2.3.0'
manga_results_basedir = os.environ['STELLARMASS_PCA_RESULTSDIR']
os.environ['STELLARMASS_PCA_CSPBASE'] = '/Users/admin/sas/mangawork/manga/mangapca/zachpace/CSPs_CKC14_MaNGA_20190215-1'
csp_basedir = os.environ['STELLARMASS_PCA_CSPBASE']
mocks_results_basedir = os.path.join(
    os.environ['STELLARMASS_PCA_RESULTSDIR'], 'mocks')

from astropy.cosmology import WMAP9
cosmo = WMAP9

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.usetex'] = True
if 'DISPLAY' not in os.environ:
    matplotlib.use('agg')
