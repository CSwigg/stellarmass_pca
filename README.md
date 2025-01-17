# stellarmass_pca

This library fits stellar-continuum observations with a basis set of `q` spectral vectors (generated from a library of synthetic star formation histories). The fit in this low-dimensional space makes various spectral-fitting degeneracies more tractable. Much of this library is a reimplimentation of [Chen et al. (2012)](http://adsabs.harvard.edu/abs/2012MNRAS.421..314C), but for [SDSS-IV/MaNGA IFS](https://www.sdss.org/surveys/manga/) data.

We will release a SDSS-IV VAC (#0060) as part of DR16 in December 2019 (link soon). This library contains some simple data-access scripting utilities for reading the data in the VAC. 

# Related Libraries
* [`cspgen`](https://github.com/zpace/cspgen), which provides some code to help you generate synthetic stellar pops
* [`fsps`](https://github.com/cconroy20/fsps), the "workhorse" of `cspgen`
* `mangapca_read` (released with SDSS DR16), a lighter-weight "data-reader"-only library intended for users who just want to read *our* analyzed prodducts

# Most Important Contents (selected)
* `read_results.py` (1): parses results from SDSS-IV Value-Added Catalog (VAC #0060)
* `find_pcs.py` (2): generates a PC vector basis set, and uses it to fit MaNGA datacubes

(1) - for user of data from VAC #0060
(2) - if you want to run your own analysis, using the provided set of synthetic training spectra

# Dependencies
* [`zpmanga`](https://github.com/zpace/zpmanga), a set of handy MaNGA-related functions
