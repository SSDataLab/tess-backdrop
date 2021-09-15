import os

import numpy as np
import pytest

import tess_backdrop as tbd
from tess_backdrop import PACKAGEDIR, __version__


def test_build():
    for cutout_size in [256]:
        fnames = ["/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/data/tempffi.fits"]
        b = tbd.FullBackDrop(
            np.hstack([fnames, fnames]),
            sector=2,
            test_frame_number=0,
            cutout_size=cutout_size,
        )
        assert b.flux(0).shape == (cutout_size, cutout_size)
        assert b.sector == 2
        assert b.camera == 1
        assert b.ccd == 3
        b.plot_test_frame()
        b.fit_model()
        b.plot(0)
        b.save(".", overwrite=True)
        b.load((2, 1, 3), dir=".")
        m = b.build_model(np.arange(30), np.arange(30))
        assert m.shape == (2, 30, 30)
