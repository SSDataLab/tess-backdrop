"""Simple version of a Backdrop """
import numpy as np
import lightkurve as lk
from scipy.sparse import csr_matrix, vstack, hstack
from lightkurve.correctors.designmatrix import _spline_basis_vector
from astropy.io import fits
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
import fitsio
from tqdm import tqdm
from functools import lru_cache

import os
from . import PACKAGEDIR
from .version import __version__
import logging


from .utils import get_spline_matrix, get_knots, get_saturation_mask

log = logging.getLogger(__name__)


@dataclass
class SimpleBackDrop(object):
    """Class to create a simple, polynomial version of a TESS BackDrop

    Simple backdrop uses only a polynomial in the spatial dimension,
    and does not account for straps or fine structure.
    """

    fnames: list  # List of file names
    nb: int = 8  # Number of bins to use to downsample images
    sector: Optional = None  # Sector (otherwise will scrape from file names)
    test_frame: Optional = None  # Reference frame

    def __post_init__(self):
        if not (np.log2(self.nb) % 1) == 0:
            raise ValueError("Number of bins must be e.g 4, 8, 16, 32")
        if (self.fnames is not None) & np.any(
            [not hasattr(self, attr) for attr in ["t_start", "camera", "ccd", "sector"]]
        ):
            self.t_start = np.zeros(self.shape[0])
            for tdx, fname in enumerate(self.fnames):
                with fits.open(fname, lazy_load_hdus=True) as hdu:
                    if fname == self.fnames[0]:
                        if self.sector is None:
                            self.sector = int(fname.split("-s")[1].split("-")[0])
                        self.camera = hdu[1].header["camera"]
                        self.ccd = hdu[1].header["ccd"]
                        self.t_start[tdx] = hdu[0].header["TSTART"]
                    else:
                        if hdu[1].header["camera"] != self.camera:
                            raise ValueError("Files must be on the same Camera")
                        if hdu[1].header["ccd"] != self.ccd:
                            raise ValueError("Files must be on the same CCD")
                        self.t_start[tdx] = hdu[0].header["TSTART"]
        s = np.argsort(self.t_start)
        if self.fnames is not None:
            self.fnames = np.asarray(self.fnames)[s]
        self.t_start = self.t_start[s]
        if self.test_frame is None:
            self.test_frame = self.shape[0] // 2
        self.sat_mask = get_saturation_mask(
            fitsio.read(self.fnames[self.test_frame])[:2048, 45 : 45 + 2048]
        ).astype(float)
        if self.ccd in [1, 3]:
            self.bore_pixel = [2048, 2048]
        elif self.ccd in [2, 4]:
            self.bore_pixel = [2048, 0]
        self.row, self.column = np.mgrid[: 2048 // self.nb, : 2048 // self.nb] * self.nb
        self.column, self.row = (self.column - self.bore_pixel[1]) / (2048), (
            self.row - self.bore_pixel[0]
        ) / (2048)
        self.rad = np.hypot(self.column, self.row) / np.sqrt(2)
        self.phi = np.arctan2(self.row, self.column)
        self.phi_knots = np.hstack(
            [
                self.phi.min(),
                self.phi.min(),
                np.linspace(self.phi.min(), self.phi.max(), 12),
                self.phi.max(),
                self.phi.max(),
            ]
        )
        self.r_knots = (
            np.hstack(
                [
                    self.rad.min(),
                    self.rad.min(),
                    np.linspace(self.rad.min(), self.rad.max(), 48),
                    self.rad.max(),
                    self.rad.max(),
                ]
            )
            ** 0.35
        )
        self.col_knots = np.hstack(
            [
                np.round(self.column.min()),
                np.linspace(
                    np.round(self.column.min()), np.round(self.column.max()), 28
                ),
                np.round(self.column.max()),
                np.round(self.column.max()),
            ]
        )
        self.row_knots = np.hstack(
            [
                np.round(self.row.min()),
                np.linspace(np.round(self.row.min()), np.round(self.row.max()), 28),
                np.round(self.row.max()),
                np.round(self.row.max()),
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
        return (s1, 2048 // self.nb, 2048 // self.nb)

    def _build_design_matrix(self):
        """Creates the design matrix of splines, based on `self.r_knots` and `self.phi_knots`"""
        # This is hard coded for now
        degree = 2

        def _X(x, knots, degree):
            matrices = [
                csr_matrix(_spline_basis_vector(x, degree, idx, knots))
                for idx in np.arange(-1, len(knots) - degree - 1)
            ]
            X = vstack(matrices, format="csr").T
            return X

        # Cartesian spline
        x1, x2 = np.round(self.column.ravel(), 10), np.round(self.row.ravel(), 10)
        knots1, knots2 = self.col_knots, self.row_knots

        X1 = _X(x1, knots1, degree)
        X2 = _X(x2, knots2, degree)
        Xf_cart = hstack([X1[:, idx].multiply(X2) for idx in np.arange(0, X1.shape[1])])

        # Radial spline
        x1, x2 = np.round(self.rad.ravel(), 10), np.round(self.phi.ravel(), 10)
        knots1, knots2 = self.r_knots, self.phi_knots

        X1 = _X(x1, knots1, degree)
        X2 = _X(x2, knots2, degree)
        Xf = hstack([X1[:, idx].multiply(X2) for idx in np.arange(0, X1.shape[1])])

        # Build matrix
        A = hstack([Xf, X1, Xf_cart, np.ones(X1.shape[0])[:, None]]).tocsr()

        self.prior_mu = np.zeros(A.shape[1]) + 2
        self.prior_sigma = np.ones(A.shape[1]) * 100
        self.design_matrix = A

    @property
    def A(self):
        """Design matrix"""
        if not hasattr(self, "design_matrix"):
            self._build_design_matrix()
        return self.design_matrix

    def __repr__(self):
        return f"SimpleBackDrop {self.shape}"

    def reshape(self, nb):
        """Reshape the corrector to a different bin size"""
        self.nb = nb
        self.__post_init__()

    def flux(self, tdx, nb=None):
        """Get the flux at a particular time index"""
        if self.fnames is None:
            return np.zeros(self.shape[1:])
        return _flux(self.fnames[tdx], [self.nb if nb is None else nb][0])

    def fit_frame(self, tdx, store=False):
        if not hasattr(self, "A"):
            self._build_design_matrix()
        if store:
            self.weights = np.isfinite(self.flux(tdx)).astype(float)
            self.weights[self.weights == 0] = 1e10
            self.sigma_w_inv = self.A.T.dot(
                self.A.multiply(1 / self.weights.ravel()[:, None] ** 2)
            ).toarray()
            self.sigma_w_inv += np.diag(1 / self.prior_sigma ** 2)
        f = np.nan_to_num(np.log10(self.flux(tdx)))
        B = (
            self.A.T.dot((f / self.weights ** 2).ravel())
            + self.prior_mu / self.prior_sigma ** 2
        )
        w = np.linalg.solve(self.sigma_w_inv, B)
        if store:
            model = 10 ** self.A.dot(w).reshape(self.shape[1:])
            self.weights[np.abs(10 ** f - model) > 300] = 1e10
            self.sigma_w_inv = self.A.T.dot(
                self.A.multiply(1 / self.weights.ravel()[:, None] ** 2)
            ).toarray()
            self.sigma_w_inv += np.diag(1 / self.prior_sigma ** 2)
        return w

    def fit_model(self, test_frame=None):
        """We use a test frame to build and store the inverse covariance matrix to make the rest of the steps faster!"""
        if test_frame is None:
            test_frame = self.test_frame
        _ = self.fit_frame(test_frame, store=True)
        self.w = np.zeros((self.shape[0], self.A.shape[1]))
        for tdx in tqdm(range(self.shape[0])):
            self.w[tdx] = self.fit_frame(tdx, store=False)

    def model(self, tdx):
        """Returns the model at a given time index"""
        if not hasattr(self, "w"):
            raise ValueError("Please fit the model with `fit_model` first")
        if not self.A.shape[0] == np.product(self.shape[1:]):
            self._build_design_matrix()
        if not self.A.shape[0] == np.product(self.shape[1:]):
            self.__post_init__()
            self._build_design_matrix()
        return 10 ** self.A.dot(self.w[tdx]).reshape(self.shape[1:])

    def _package_weights_hdulist(self):
        """Put the masks into a fits format
        Save a model fit to the tess-backrop data directory.

        Will create a fits file containing the following extensions
            - Primary
            - T_START: The time array for each background solution
            - KNOTS: Knot spacing in row and column
            - SPLINE_W: Solution to the spline model. Has shape (ntimes x nknots x nknots)
            - STRAP_W: Solution to the strap model. Has shape (ntimes x self.cutout_size)
            - POLY_W: Solution to the polynomial model. Has shape (ntimes x npoly x npoly)
        """
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
        hdu1 = fits.BinTableHDU.from_columns(cols, name="QUALITY")
        cols = [fits.Column(name="R_KNOTS", format="D", unit="PIX", array=self.r_knots)]
        hdu2 = fits.BinTableHDU.from_columns(cols, name="R_KNOTS")
        cols = [
            fits.Column(name="phi_KNOTS", format="D", unit="PIX", array=self.phi_knots)
        ]
        hdu3 = fits.BinTableHDU.from_columns(cols, name="PHI_KNOTS")

        hdu4 = fits.ImageHDU(self.w[s], name="weights")
        hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4])
        return hdul

    def save(self, output_dir=None, overwrite=False):
        """Save fits files to `output_dir`"""
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

    def load(self, sector, camera, ccd, nb=1):
        """
        Load a model fit to the tess-backrop data directory.

        Parameters
        ----------
        sector: int
            TESS sector number
        camera: int
            TESS camera number
        ccd: int
            TESS CCD number
        """
        dir = f"{PACKAGEDIR}/data/sector{sector:03}/camera{camera:02}/ccd{ccd:02}/"
        if not os.path.isdir(dir):
            raise ValueError(
                f"No solutions exist for Sector {sector}, Camera {camera}, CCD {ccd}."
            )
        fname = f"tessbackdrop_simple_sector{sector}_camera{camera}_ccd{ccd}.fits"
        with fits.open(dir + fname, lazy_load_hdus=True) as hdu:
            for key in ["sector", "camera", "ccd", "nb"]:
                setattr(self, key, hdu[0].header[key])
            self.t_start = hdu[1].data["T_START"]
            if "QUALITY" in hdu[1].data.names:
                self.quality = hdu[1].data["QUALITY"]
            self.r_knots = hdu[2].data["R_KNOTS"]
            self.phi_knots = hdu[3].data["PHI_KNOTS"]
            self.w = hdu[4].data

    def build_model(self, column, row, times=None, poly_only=False):
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
        poly_only: bool
            Whether to return just the polynomial terms, (i.e. no splines, no straps)
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
        self._build_design_matrix()
        bkg = np.zeros((len(tdxs), *self.row.shape))
        for idx, tdx in enumerate(tdxs):
            bkg[idx, :, :] = 10 ** self.A.dot(self.w[tdx]).reshape(self.row.shape)
        return bkg

    def correct_tpf(self, tpf, exptime=None):
        """Returns a TPF with the background corrected

        Parameters
        ----------
        tpf : lk.TargetPixelFile
            Target Pixel File object. Must be a TESS target pixel file, and must
            be a 30 minute cadence.
        exptime : float, None
            The exposure time between each cadence. If None, will be generated from the data

        Returns
        -------
        corrected_tpf : lk.TargetPixelFile
            New TPF object, with the TESS background removed.
        """
        if exptime is None:
            exptime = np.median(np.diff(tpf.time.value))
        if exptime < 0.02:
            raise ValueError(
                "tess_backdrop can only correct 30 minute cadence FFIs currently."
            )
        if tpf.mission.lower() != "tess":
            raise ValueError("tess_backdrop can only correct TESS TPFs.")

        if not hasattr(self, "sector"):
            self.load(sector=tpf.sector, camera=tpf.camera, ccd=tpf.ccd)
        else:
            if (
                (self.sector != tpf.sector)
                | (self.camera != tpf.camera)
                | (self.ccd != tpf.ccd)
            ):
                self.load(sector=tpf.sector, camera=tpf.camera, ccd=tpf.ccd)

        tdxs = [
            np.argmin(np.abs((self.t_start - t) + exptime))
            for t in tpf.time.value
            if (np.min(np.abs((self.t_start - t) + exptime)) < exptime)
        ]

        bkg = self.build_model(
            np.arange(tpf.shape[2]) + tpf.column,
            np.arange(tpf.shape[1]) + tpf.row,
            times=tdxs,
        )
        return tpf - bkg

    def plot(self, frame=0):
        """Plots a given frame of the data, model and residuals"""
        with plt.style.context("seaborn-white"):
            f = self.flux(frame)
            if np.nansum(f) == 0:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True, sharey=True)
                v = np.nanpercentile(self.model(frame), [1, 99])
                ax = [ax]
            else:
                fig, ax = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
                v = np.nanpercentile(f, [1, 99])
            ax[0].imshow(self.model(frame), vmin=v[0], vmax=v[1], cmap="Greys_r")
            ax[0].set(xlabel="Column", ylabel="Row", title="Model")
            if np.nansum(f) == 0:
                return ax[0]
            ax[1].imshow(f, vmin=v[0], vmax=v[1], cmap="Greys_r")
            ax[1].set(xlabel="Column", title="Data")
            if np.nansum(f) != 0:
                ax[2].imshow(
                    self.flux(frame) - self.model(frame),
                    vmin=-10,
                    vmax=10,
                    cmap="coolwarm",
                )
                ax[2].set(xlabel="Column", title="Residuals")
        return ax


@lru_cache(maxsize=16)
def _flux(fname, nb):
    """Let's us cache some frames in memory"""
    ar = np.zeros((2048 // nb, 2048 // nb, nb, nb))
    flux1 = fitsio.read(fname)[:2048, 45 : 45 + 2048]
    for idx in range(nb):
        for jdx in range(nb):
            ar[:, :, idx, jdx] = flux1[idx::nb, jdx::nb]
    ar[ar <= 0] = np.nan
    return np.min(ar, axis=(2, 3))
