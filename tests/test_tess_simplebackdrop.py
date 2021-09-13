import numpy as np

import os
import pytest
import tess_backdrop as tbd
from tess_backdrop import PACKAGEDIR
from tess_backdrop import __version__


def test_build():
    for cutout_size in [2048, 512]:
        fnames = ["/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/data/tempffi.fits"]
        b = tbd.SimpleBackDrop(
            np.hstack([fnames, fnames]), sector=2, test_frame=0, cutout_size=cutout_size
        )
        if cutout_size == 2048:
            assert b.flux(0).shape == (256, 256)
        if cutout_size == 512:
            assert b.flux(0).shape == (64, 64)
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
