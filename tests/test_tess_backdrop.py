import numpy as np

import os
import pytest
import tess_backdrop as tbd
from tess_backdrop import PACKAGEDIR
from tess_backdrop import __version__


def is_action():
    try:
        return os.environ["GITHUB_ACTIONS"]
    except KeyError:
        return False


def test_version():
    assert __version__ == "0.1.4"
    print(is_action())


@pytest.mark.skipif(
    is_action(), reason="Can not run on GitHub actions, because files too large."
)
def test_load():
    b = tbd.BackDrop()
    b.load(1, 1, 3)
    assert b.sector == 1
    assert b.camera == 1
    assert b.ccd == 3

    corr = b.build_correction(np.arange(100, 110), np.arange(200, 210), 0)
    assert len(corr.shape) == 3
    assert corr.shape == (1, 10, 10)

    corr = b.build_correction(np.arange(100, 101), np.arange(200, 201))
    assert len(corr.shape) == 3
    assert corr.shape == (1282, 1, 1)


def test_build():

    fnames = [
        "/".join(PACKAGEDIR.split("/")[:-2])
        + "/tests/data/tess2018206192942-s0001-1-4-0120-s_ffic.fits"
    ]
    cutout_size = 128
    b = tbd.BackDrop(fnames, cutout_size=cutout_size, nknots=5)
    b.fit_model()
