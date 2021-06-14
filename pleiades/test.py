"""Run script for pleiades"""
import numpy as np
import tess_backdrop as tbd
from glob import glob
import sys

sectors = np.arange(1, 14)
cameras = np.arange(1, 5)
ccds = np.arange(1, 5)

dict = {
    idx + (4 * jdx) + (16 * kdx): (sector, camera, ccd)
    for kdx, sector in enumerate(sectors)
    for jdx, camera in enumerate(cameras)
    for idx, ccd in enumerate(ccds)
}

# dir = "/Volumes/Nibelheim/tess/"
dir = "/nobackupp12/chedges/tess/"


def get_fits_file(idx):
    sector, camera, ccd = dict[idx]
    dirname = dir + f"sector{sector:02}/"
    if len(glob(dirname + f"camera{camera:02}")) != 0:
        dirname += f"camera{camera:02}/"
    if len(glob(dirname + f"camera{camera}")) != 0:
        dirname += f"camera{camera}/"
    if len(glob(dirname + f"ccd{ccd:02}")) != 0:
        dirname += f"ccd{ccd:02}/"
    if len(glob(dirname + f"ccd{ccd}")) != 0:
        dirname += f"ccd{ccd}/"
    fnames = glob(dirname + f"tess*s{sector:04}-{camera}-{ccd}*_ffic.fits*")
    if len(fnames) == 0:
        return 0
    print(f"Running s{sector} c{camera} ccd{ccd}")
    b = tbd.BackDrop(fnames=fnames)
    b.fit_model()
    b.save()
    print(f"s{sector} c{camera} ccd{ccd} finished")


if __name__ == "__main__":
    if isinstance(sys.argv[1], str):
        try:
            idx = int(sys.argv[1])
        except ValueError:
            raise ValueError(f"Can not parse {sys.argv[1]} as an integer.")
    # seq 4 indexes from 1.
    get_fits_file(idx - 1)
