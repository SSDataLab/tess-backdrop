import numpy as np

import tess_backdrop as tbd
from tess_backdrop import PACKAGEDIR
from tess_backdrop import __version__


def test_version():
    assert __version__ == "0.1.2"


def test_load():
    b = tbd.BackDrop()
    b.load(2, 1, 3)
    assert b.sector == 2
    assert b.camera == 1
    assert b.ccd == 3

    corr = b.build_correction(np.arange(100, 110), np.arange(200, 210), 0)
    assert len(corr.shape) == 3
    assert corr.shape == (1, 10, 10)

    corr = b.build_correction(np.arange(100, 101), np.arange(200, 201))
    assert len(corr.shape) == 3
    assert corr.shape == (1245, 1, 1)


def test_build():
    fnames = [
        "/".join(PACKAGEDIR.split("/")[:-2])
        + "/tests/data/tess2018206192942-s0001-1-4-0120-s_ffic.fits"
    ]
    cutout_size = 128
    b = tbd.BackDrop(fnames, cutout_size=cutout_size, nknots=5)
    b.fit_model()
