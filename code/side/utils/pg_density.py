# density and cdf of the polya-gamma distribution PG(1, 0)
#
# no closed form, but two alternating series: one converges fast for large
# omega, the other for small omega. we switch between them at 0.15

import numpy as np
from scipy.special import erfc

_K = np.arange(30)[:, None]          # series terms, as a column for broadcasting
_SIGNS = (-1.0) ** _K

def _pdf_large(o):
    return 2 * np.pi * (_SIGNS * (2*_K + 1) * np.exp(-(2*_K + 1)**2 * np.pi**2 * o / 2)).sum(0)

def _pdf_small(o):
    return (_SIGNS * (2*_K + 1) * np.exp(-(2*_K + 1)**2 / (8 * o))).sum(0) / np.sqrt(2 * np.pi * o**3)

def _cdf_large(o):
    return 1 - (_SIGNS * 4 / ((2*_K + 1) * np.pi) * np.exp(-(2*_K + 1)**2 * np.pi**2 * o / 2)).sum(0)

def _cdf_small(o):
    return 2 * (_SIGNS * erfc((2*_K + 1) / np.sqrt(8 * o))).sum(0)

# the density is 0 at omega = 0 (the small omega series is 0/0 there)
def pg_pdf(omega):
    omega = np.asarray(omega, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(omega <= 0, 0.0, np.where(omega < 0.15, _pdf_small(omega), _pdf_large(omega)))

def pg_cdf(omega):
    omega = np.asarray(omega, dtype=float)
    return np.where(omega < 0.15, _cdf_small(omega), _cdf_large(omega))
