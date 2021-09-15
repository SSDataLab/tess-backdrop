"""Simple version of a Backdrop """
import logging
import os
from dataclasses import dataclass
from typing import Optional

import fitsio
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from lightkurve.correctors.designmatrix import _spline_basis_vector
from scipy.sparse import csr_matrix, hstack, vstack
from tqdm import tqdm

from . import PACKAGEDIR
from .utils import (
    get_knots,
    get_saturation_mask,
    get_spline_matrix,
    _flux,
    _X,
    _bin_down,
    plot,
)
from .version import __version__

log = logging.getLogger(__name__)


@dataclass
class SimpleBackDrop(object):
    """Class to create a simple, polynomial version of a TESS BackDrop

    Simple backdrop uses only a polynomial in the spatial dimension, and a spline
    in the radial dimension. SimpleBackDrop does not account for straps or fine structure
    and does not produce corrections for jitter terms.

    SimpleBackDrop corrector bins the data down. The default is to bin each dimension
    by a factor of 8, i.e. to reduce the 2048x2048 images to 256x256 images. This
    makes it faster and cheaper to make the corrections. In each bin, we take the
    minimum pixel value, meaning that we're always attempting to fit the background
    and not the targets.

    Parameters
    ----------
    fnames : Optional str
        List of file names

    """

    fnames: Optional = None  # List of file names
    nb: int = 8  # Number of bins to use to downsample images
    npoly: int = 6  # Polynomial order for cartesian
    sector: Optional = None  # Sector (otherwise will scrape from file names)
    test_frame_number: Optional = None  # Reference frame
    cutout_size: int = 2048  # Size to cut out for faster run time (testing only)
    njitter: int = 5000  # Number of jitter components

    def __post_init__(self):
        if not (np.log2(self.nb) % 1) == 0:
            raise ValueError("Number of bins must be e.g 4, 8, 16, 32")
        if not (np.log2(self.cutout_size) % 1) == 0:
            raise ValueError("`cutout_size`, must be e.g. 2048, 1024, 512, 256 etc")
        if (self.fnames is not None) & np.any(
            [not hasattr(self, attr) for attr in ["t_start", "camera", "ccd", "sector"]]
        ):
            self.t_start, self.quality = np.zeros((2, self.shape[0]))
            for tdx, fname in enumerate(self.fnames):
                with fits.open(fname, lazy_load_hdus=True) as hdu:
                    if fname == self.fnames[0]:
                        if self.sector is None:
                            try:
                                self.sector = int(fname.split("-s")[1].split("-")[0])
                            except:
                                raise ValueError(
                                    "Can not parse file name for sector number"
                                )
                        self.camera = hdu[1].header["camera"]
                        self.ccd = hdu[1].header["ccd"]
                        self.t_start[tdx] = hdu[0].header["TSTART"]
                        self.quality[tdx] = hdu[1].header["DQUALITY"]
                    else:
                        if hdu[1].header["camera"] != self.camera:
                            raise ValueError("Files must be on the same Camera")
                        if hdu[1].header["ccd"] != self.ccd:
                            raise ValueError("Files must be on the same CCD")
                        self.t_start[tdx] = hdu[0].header["TSTART"]
                        self.quality[tdx] = hdu[1].header["DQUALITY"]
        if (self.test_frame_number is None) and (self.fnames is not None):
            try:
                # Test frame is a low flux frame, close to the middle of the dataset.
                f = np.asarray(
                    [
                        fitsio.read(fname)[
                            np.min([self.bore_pixel[0], 2047]),
                            45 + np.min([self.bore_pixel[1], 0]),
                        ]
                        for fname in self.fnames
                    ]
                )

                l = np.where((f < np.nanpercentile(f, 5)) & (self.quality == 0))[0]
                self.test_frame_number = l[np.argmin(np.abs(l - len(f) // 2))]
            except:
                self.test_frame_number = self.shape[0] // 2

        if self.fnames is not None:
            s = np.argsort(self.t_start)
            self.fnames = np.asarray(self.fnames)[s]
            self.t_start = self.t_start[s]

        if hasattr(self, "ccd"):
            if self.ccd in [1, 3]:
                self.bore_pixel = [2048, 2048]
            elif self.ccd in [2, 4]:
                self.bore_pixel = [2048, 0]
            self.row, self.column = (
                np.mgrid[: self.cutout_size // self.nb, : self.cutout_size // self.nb]
                * self.nb
            )
            self.column, self.row = (self.column - self.bore_pixel[1]) / (2048), (
                self.row - self.bore_pixel[0]
            ) / (2048)
            self.rad = np.hypot(self.column, self.row) / np.sqrt(2)
            self.phi = np.arctan2(self.row, self.column)
            if self.ccd in [2, 4]:
                self.phi_knots = np.pi * np.hstack(
                    [
                        -0.5,
                        -0.5,
                        np.linspace(-0.5, 0, 8),
                        0,
                        0,
                    ]
                )
            if self.ccd in [1, 3]:
                self.phi_knots = np.pi * np.hstack(
                    [
                        -1,
                        -1,
                        np.linspace(-1, 0.5, 8),
                        -0.5,
                        -0.5,
                    ]
                )
            self.r_knots = np.hstack(
                [
                    0,
                    0,
                    np.linspace(0, 1, 58) ** 0.5,
                    1,
                    1,
                ]
            )

    @property
    def shape(self):
        if self.fnames is not None:
            s1 = len(self.fnames)
        elif hasattr(self, "w"):
            s1 = len(self.w)
        elif hasattr(self, "t_start"):
            s1 = len(self.t_start)
        else:
            raise ValueError("Unknown shape?")
        return (s1, self.cutout_size // self.nb, self.cutout_size // self.nb)

    def _simple_design_matrix(self):
        """Creates the design matrix of splines, based on `self.r_knots` and `self.phi_knots`"""
        X1 = _X(self.rad.ravel(), self.r_knots, 2)
        X2 = _X(self.phi.ravel(), self.phi_knots, 2)

        Xf = hstack([X1[:, idx].multiply(X2) for idx in np.arange(8, X1.shape[1])])
        Xf = hstack([X1[:, :8], Xf])

        poly = [
            self.column.ravel() ** idx * self.row.ravel() ** jdx
            for idx in np.arange(0, self.npoly + 1)
            for jdx in np.arange(0, self.npoly + 1)
        ]
        Xf_poly = np.vstack(poly).T

        # Build matrix
        A = hstack(
            [
                Xf,
                Xf_poly,
            ]
        ).tocsr()
        return A

    def _build_simple_design_matrix(self):
        """Builds the design matrix for use"""
        A = self._simple_design_matrix()
        self.simple_prior_mu = np.zeros(A.shape[1]) + 2
        self.simple_prior_sigma = np.ones(A.shape[1]) * 100
        self.simple_design_matrix = A

    @property
    def A(self):
        """Design matrix, this property helps us write some math"""
        if not hasattr(self, "simple_design_matrix"):
            self._build_simple_design_matrix()
        return self.simple_design_matrix

    def __repr__(self):
        try:
            return f"SimpleBackDrop {self.shape}"
        except ValueError:
            return f"SimpleBackDrop"

    def reshape(self, nb):
        """Reshape the corrector to a different bin size"""
        self.nb = nb
        self.__post_init__()

    def flux(self, tdx, nb=None):
        """Get the flux at a particular time index"""
        if self.fnames is None:
            return np.zeros(self.shape[1:])
        if not hasattr(self, "star_mask"):
            self.star_mask = (
                np.hypot(
                    *np.gradient(
                        np.nan_to_num(
                            _flux(
                                self.fnames[self.test_frame_number],
                                cutout_size=self.cutout_size,
                            )
                        )
                    )
                )
                > 30
            )
            self.star_mask[:-1] |= self.star_mask[1:]
            self.star_mask[1:] |= self.star_mask[:-1]
            self.star_mask[:, :-1] |= self.star_mask[:, 1:]
            self.star_mask[:, 1:] |= self.star_mask[:, :-1]
            self.star_mask = (~self.star_mask).astype(float)
            self.star_mask[self.star_mask == 0] = np.nan

        if not hasattr(self, "sat_mask"):
            self.sat_mask = (
                get_saturation_mask(
                    fitsio.read(self.fnames[self.test_frame_number])[
                        : self.cutout_size, 45 : 45 + self.cutout_size
                    ],
                    cutout_size=self.cutout_size,
                )
            ).astype(float)
            self.sat_mask[self.sat_mask == 0] = np.nan

        if not hasattr(self, "jitter_mask"):
            l1, l2 = np.where(~np.isfinite(self.star_mask) & np.isfinite(self.sat_mask))
            s = np.random.choice(len(l1), size=self.njitter, replace=False)
            l1, l2 = l1[s], l2[s]
            self.jitter_mask = np.zeros((self.cutout_size, self.cutout_size), bool)
            self.jitter_mask[l1, l2] = True

        return _bin_down(
            _flux(self.fnames[tdx], cutout_size=self.cutout_size)
            * self.sat_mask
            * self.star_mask,
            [self.nb if nb is None else nb][0],
            cutout_size=self.cutout_size,
        )

    def fit_frame(self, tdx, store=False):
        """Fit an individual frame of the stack."""
        if not hasattr(self, "A"):
            self._build_simple_design_matrix()
        if store:
            self.weights = np.isfinite(self.flux(tdx))
            self.weights &= self.flux(tdx) > 10
            self.weights = self.weights.astype(float)
            self.weights[self.weights == 0] = 1e10
            self.sigma_w_inv = self.A.T.dot(
                self.A.multiply(1 / self.weights.ravel()[:, None] ** 2)
            ).toarray()
            self.sigma_w_inv += np.diag(1 / self.simple_prior_sigma ** 2)
        f = np.nan_to_num(np.log10(self.flux(tdx)))
        B = (
            self.A.T.dot((f / self.weights ** 2).ravel())
            + self.simple_prior_mu / self.simple_prior_sigma ** 2
        )
        w = np.linalg.solve(self.sigma_w_inv, B)
        if store:
            model = 10 ** self.A.dot(w).reshape(self.shape[1:])
            self.weights[np.abs(10 ** f - model) > 20] = 1e10
            self.sigma_w_inv = self.A.T.dot(
                self.A.multiply(1 / self.weights.ravel()[:, None] ** 2)
            ).toarray()
            self.sigma_w_inv += np.diag(1 / self.simple_prior_sigma ** 2)
        return w

    def fit_model(self, test_frame_number=None):
        """Fit the Backdrop model to the image stack in find_bad_frames

        Parameters
        ----------
            test_frame_number : int
                Optional frame to use as a test frame
        """
        if test_frame_number is None:
            test_frame_number = self.test_frame_number
        _ = self.fit_frame(test_frame_number, store=True)
        self.w = np.zeros((self.shape[0], self.A.shape[1]))
        for tdx in tqdm(range(self.shape[0])):
            self.w[tdx] = self.fit_frame(tdx, store=False)

    def model(self, tdx):
        """Returns the model at a given time index"""
        if not hasattr(self, "w"):
            raise ValueError("Please fit the model with `fit_model` first")
        if not self.A.shape[0] == np.product(self.shape[1:]):
            self._build_simple_design_matrix()
        if not self.A.shape[0] == np.product(self.shape[1:]):
            self.__post_init__()
            self._build_simple_design_matrix()
        return 10 ** self.A.dot(self.w[tdx]).reshape(self.shape[1:])

    def _package_weights_hdulist(self):
        """Put the masks into a fits format"""
        hdu0 = self.hdu0
        s = np.argsort(self.t_start)
        cols = [
            fits.Column(
                name="T_START", format="D", unit="BJD - 2457000", array=self.t_start[s]
            )
        ]
        if hasattr(self, "quality"):
            cols.append(
                fits.Column(
                    name="QUALITY",
                    format="D",
                    unit="BJD - 2457000",
                    array=self.quality[s],
                )
            )
        hdu1 = fits.BinTableHDU.from_columns(cols, name="TIME")
        cols = [fits.Column(name="R_KNOTS", format="D", unit="PIX", array=self.r_knots)]
        hdu2 = fits.BinTableHDU.from_columns(cols, name="R_KNOTS")
        cols = [
            fits.Column(name="PHI_KNOTS", format="D", unit="PIX", array=self.phi_knots)
        ]
        hdu3 = fits.BinTableHDU.from_columns(cols, name="PHI_KNOTS")
        hdu4 = fits.ImageHDU(self.w[s], name="simple_weights")
        hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4])
        return hdul

    def save(self, output_dir=None, overwrite=False):
        """Save a model fit to the tess-backrop data directory.

        Will create a fits file containing the following extensions
            - Primary
            - TIME: The time array for each background solution
            - R_KNOTS: Knot spacing in radial dimension
            - PHI_KNOTS: Knot spacing in phi dimension
            - SIMPLE_WEIGHTS: Weights for the model.
        """
        if not hasattr(self, "w"):
            raise ValueError("Please fit the model with `fit_model` first")
        self.hdu0 = fits.PrimaryHDU()
        self.hdu0.header["ORIGIN"] = "tess-backdrop"
        self.hdu0.header["AUTHOR"] = "christina.l.hedges@nasa.gov"
        self.hdu0.header["VERSION"] = __version__
        for key in [
            "sector",
            "camera",
            "ccd",
            "nb",
            "npoly",
            "njitter",
            "test_frame_number",
            "cutout_size",
        ]:
            self.hdu0.header[key] = getattr(self, key)

        if output_dir is None:
            output_dir = f"{PACKAGEDIR}/data/sector{self.sector:03}/camera{self.camera:02}/ccd{self.ccd:02}/"
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

        log.info(f"Saving s{self.sector} c{self.camera} ccd{self.ccd}")
        hdul = self._package_weights_hdulist()
        fname = f"tessbackdrop_simple_sector{self.sector}_camera{self.camera}_ccd{self.ccd}.fits"
        hdul.writeto(output_dir + fname, overwrite=overwrite)

    def load(self, input, dir=None):
        """
        Load a model fit to the tess-backrop data directory.

        Parameters
        ----------
        input: tuple or string
            Either pass a tuple with `(sector, camera, ccd)` or pass
            a file name in `dir` to load
        dir : str
            Optional tring with the directory name

        Returns
        -------
        self: `tess_backdrop.SimpleBackDrop` object
        """
        if isinstance(input, tuple):
            if len(input) == 3:
                sector, camera, ccd = input
                fname = (
                    f"tessbackdrop_simple_sector{sector}_camera{camera}_ccd{ccd}.fits"
                )
            else:
                raise ValueError("Please pass tuple as `(sector, camera, ccd)`")
        elif isinstance(input, str):
            fname = input
        else:
            raise ValueError("Can not parse input")
        if dir is None:
            dir = f"{PACKAGEDIR}/data/sector{sector:03}/camera{camera:02}/ccd{ccd:02}/"
        if not os.path.isdir(dir):
            raise ValueError(f"No solutions exist")

        with fits.open(dir + fname, lazy_load_hdus=True) as hdu:
            for key in [
                "sector",
                "camera",
                "ccd",
                "nb",
                "npoly",
                "njitter",
                "test_frame_number",
                "cutout_size",
            ]:
                setattr(self, key, hdu[0].header[key])
            self.t_start = hdu[1].data["T_START"]
            if "QUALITY" in hdu[1].data.names:
                self.quality = hdu[1].data["QUALITY"]
            self.r_knots = hdu[2].data["R_KNOTS"]
            self.phi_knots = hdu[3].data["PHI_KNOTS"]
            self.w = hdu[4].data
        self.__post_init__()
        return self

    def build_model(self, column, row, times=None):
        """Build a background correction for a given column, row and time array.

        Parameters
        ----------
        column : 1D np.ndarray of ints
            Array between 0 and 2048 indicating the column number.
            NOTE: Columns in TESS FFIs and TPFs are offset by 45 pixels, and usually
            are between 45 and 2093.
        row : 1D np.ndarray of ints
            Array between 0 and 2048 indicating the row number.
        times: None, list of ints, or np.ndarray of floats
            Times to evaluate the background model at. If none, will evaluate at all
            the times for available FFIs. If array of ints, will use those indexes to the original FFIs.
            Otherwise, must be an np.ndarray of floats for the T_START time of the FFI.
        Returns
        -------
        bkg : np.ndarray
            2D array with shape ntimes x nrow x ncolumn containing the background
            estimate for the input column, row and times.
        """
        if not hasattr(self, "w"):
            raise ValueError("Please run or load a model.")

        if times is None:
            tdxs = np.arange(len(self.t_start))
        else:
            if not hasattr(times, "__iter__"):
                times = [times]
            if np.all([isinstance(i, (int, np.int64)) for i in times]):
                tdxs = times
            elif not np.in1d(np.round(times, 6), np.round(self.t_start, 6)).all():
                raise ValueError(
                    "tess-backdrop can not estimate some times in the input `times` array."
                    "No background information at that time."
                )
            else:
                tdxs = np.asarray(
                    [
                        np.where(np.round(self.t_start, 6) == np.round(t, 6))[0][0]
                        for t in times
                    ]
                )

        c, r = np.meshgrid(column, row)
        self.column, self.row = (c - self.bore_pixel[1]) / (2048), (
            r - self.bore_pixel[0]
        ) / (2048)
        self.rad = np.hypot(self.column, self.row) / np.sqrt(2)
        self.phi = np.arctan2(self.row, self.column)
        self._build_simple_design_matrix()
        bkg = np.zeros((len(tdxs), *self.row.shape))
        for idx, tdx in enumerate(tdxs):
            bkg[idx, :, :] = 10 ** self.A.dot(self.w[tdx]).reshape(self.row.shape)
        return bkg

    def correct_tpf(self, tpf, exptime=None):
        return correct_tpf(self, tpf, exptime=None)

    def plot(self, frame=0, vmin=None, vmax=None):
        """Plots a given frame of the data, model and residuals"""
        return plot(self, frame=frame, vmin=vmin, vmax=vmax)

    def plot_test_frame(self):
        if self.fnames is None:
            raise ValueError("No file names to plot")
        with plt.style.context("seaborn-white"):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True, sharey=True)
            ax = [ax]
            f = _flux(self.fnames[self.test_frame_number], cutout_size=self.cutout_size)
            im = ax[0].imshow(
                f,
                vmin=np.nanpercentile(f, 10),
                vmax=np.nanpercentile(f, 90),
                cmap="viridis",
            )
            cbar = plt.colorbar(im, ax=ax[0])
            cbar.set_label("Counts")
            ax[0].set(title="Test Frame", xlabel="Column", ylabel="Row")
        return ax
