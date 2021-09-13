import lightkurve as lk
from dataclasses import dataclass
from glob import glob
import numpy as np
from scipy.sparse import hstack, vstack, lil_matrix, csr_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.io import fits


import os
from .simple import SimpleBackDrop, _X, _flux
from .version import __version__
from . import PACKAGEDIR
from typing import Optional
from dataclasses import dataclass

import logging

log = logging.getLogger(__name__)


@dataclass
class FullBackDrop(object):
    fnames: Optional = None  # List of file names
    nb: int = 8  # Number of bins to use to downsample images
    npoly: int = 6  # Polynomial order for cartesian
    nknots: int = 30  # Number of knots for spline
    degree: int = 2  # Spline degree
    sector: Optional = None  # Sector (otherwise will scrape from file names)
    test_frame: Optional = None  # Reference frame
    cutout_size: Optional = 2048
    njitter: int = 5000  # Number of jitter components

    def __post_init__(self):
        if not (np.log2(self.cutout_size) % 1) == 0:
            raise ValueError("`cutout_size`, must be e.g. 2048, 1024, 512, 256 etc")
        self._simplebackdrop = SimpleBackDrop(
            fnames=self.fnames,
            nb=1,
            npoly=self.npoly,
            sector=self.sector,
            test_frame=self.test_frame,
            cutout_size=self.cutout_size,
            njitter=self.njitter,
        )
        self.sector = self._simplebackdrop.sector
        if hasattr(self._simplebackdrop, "test_frame"):
            self.test_frame = self._simplebackdrop.test_frame
        if hasattr(self._simplebackdrop, "ccd"):
            self.column, self.row = (
                np.arange(self.cutout_size) - self.bore_pixel[1]
            ) / 2048, ((np.arange(self.cutout_size) - self.bore_pixel[0])) / (2048)

    @property
    def row_knots_wbounds(self):
        m1, m2 = (0 - self.bore_pixel[0]) / 2048, (
            self.cutout_size - self.bore_pixel[0]
        ) / 2048
        return np.hstack([[m1, m1], np.linspace(m1, m2, self.nknots), [m2, m2]])

    @property
    def column_knots_wbounds(self):
        m1, m2 = (0 - self.bore_pixel[1]) / 2048, (
            self.cutout_size - self.bore_pixel[1]
        ) / 2048
        return np.hstack([[m1, m1], np.linspace(m1, m2, self.nknots), [m2, m2]])

    #        self._build_full_matrices()

    def __repr__(self):
        try:
            return f"FullBackDrop {self.shape}"
        except ValueError:
            return f"FullBackDrop"

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
        return (s1, self.cutout_size, self.cutout_size)

    @property
    def t_start(self):
        return self._simplebackdrop.t_start

    @property
    def quality(self):
        return self._simplebackdrop.quality

    @property
    def ccd(self):
        return self._simplebackdrop.ccd

    @property
    def bore_pixel(self):
        return self._simplebackdrop.bore_pixel

    @property
    def camera(self):
        return self._simplebackdrop.camera

    @property
    def star_mask(self):
        return self._simplebackdrop.star_mask

    @property
    def sat_mask(self):
        return self._simplebackdrop.sat_mask

    @property
    def jitter_mask(self):
        return self._simplebackdrop.jitter_mask

    def plot_test_frame(self, **kwargs):
        self._simplebackdrop.plot_test_frame(**kwargs)

    def _spline_design_matrix(self, xc=None, xr=None):
        if xc is None:
            xc = self.column
        if xr is None:
            xr = self.row

        Xc = _X(xc, self.column_knots_wbounds, 2)
        Xr = _X(xr, self.row_knots_wbounds, 2)
        Xcf = vstack([Xc for idx in range(len(xr))]).tocsr()
        Xrf = (
            hstack([Xr for idx in range(len(xc))])
            .reshape((Xcf.shape[0], Xc.shape[1]))
            .tocsr()
        )
        Xf = hstack([Xrf.multiply(X.T) for X in Xcf.T]).tocsr()
        return Xf.tocsr()

    def _strap_design_matrix(self):
        e = lil_matrix((self.cutout_size, self.cutout_size ** 2))
        for idx in range(self.cutout_size):
            e[idx, np.arange(self.cutout_size) * self.cutout_size + idx] = 1
        return e.T.tocsr()

    def _build_spline_design_matrix(self, xc=None, xr=None):
        """Builds the design matrix for use"""
        self.spline_prior_mu = np.zeros((self.nknots + 2) ** 2)
        self.spline_prior_sigma = np.ones((self.nknots + 2) ** 2) * 300
        self.spline_design_matrix = self._spline_design_matrix(xc, xr)

    def _build_strap_design_matrix(self):
        """Builds the design matrix for use"""
        self.strap_prior_mu = np.zeros(self.cutout_size)
        self.strap_prior_sigma = np.ones(self.cutout_size) * 300
        self.strap_design_matrix = self._strap_design_matrix()

    def _build_full_matrices(self):
        self._simplebackdrop.reshape(1)
        self.simple_design_matrix = self._simplebackdrop._simple_design_matrix()
        self._simplebackdrop.reshape(self.nb)

        self._build_spline_design_matrix()
        self._build_strap_design_matrix()

        self.A = hstack([self.spline_design_matrix, self.strap_design_matrix])
        self.prior_mu = np.hstack([self.spline_prior_mu, self.strap_prior_mu])
        self.prior_sigma = np.hstack([self.spline_prior_sigma, self.strap_prior_sigma])

    def flux(self, tdx, nb=None):
        """Get the flux at a particular time index"""
        if self.fnames is None:
            return np.zeros(self.shape[1:])
        return _flux(self.fnames[tdx], cutout_size=self.cutout_size)

    def fit_frame(self, tdx, store=False):
        """Fit an individual frame of the stack."""
        simple_w = self._simplebackdrop.fit_frame(tdx, store=store)
        if store:
            # We store the sigma_w_inv because it's computationally expensive to invert
            # and we don't need to update the weights
            self.weights = self.sat_mask * self.star_mask
            self.weights[~np.isfinite(self.weights)] = 1e10
            self.sigma_w_inv = self.A.T.dot(
                self.A.multiply(1 / self.weights.ravel()[:, None] ** 2)
            ).toarray()
            self.sigma_w_inv += np.diag(1 / self.prior_sigma ** 2)

        simple_model = 10 ** self.simple_design_matrix.dot(simple_w).reshape(
            self.shape[1:]
        )
        f = np.nan_to_num(self.flux(tdx) - simple_model)
        B = (
            self.A.T.dot((f / self.weights ** 2).ravel())
            + self.prior_mu / self.prior_sigma ** 2
        )
        w = np.linalg.solve(self.sigma_w_inv, B)
        jitter = (f - self.A.dot(w).reshape(self.shape[1:]))[self.jitter_mask]
        return (
            w[: (self.nknots + 2) ** 2],
            w[(self.nknots + 2) ** 2 :],
            simple_w,
            jitter,
        )

    def fit_model(self, test_frame=None):
        """Fit the backdrop model to the image stack"""
        if test_frame is None:
            test_frame = self.test_frame
        self._build_full_matrices()
        _ = self.fit_frame(test_frame, store=True)
        self.spline_w = np.zeros((self.shape[0], self.spline_design_matrix.shape[1]))
        self.strap_w = np.zeros((self.shape[0], self.strap_design_matrix.shape[1]))
        self.jitter = np.zeros((self.shape[0], self.njitter))
        self._simplebackdrop.w = np.zeros(
            (self.shape[0], self.simple_design_matrix.shape[1])
        )
        for tdx in tqdm(range(self.shape[0])):
            (
                self.spline_w[tdx],
                self.strap_w[tdx],
                self._simplebackdrop.w[tdx],
                self.jitter[tdx],
            ) = self.fit_frame(tdx, store=False)

    def model_simple(self, tdx):
        """Returns the simple model at a given time index"""
        return 10 ** self.simple_design_matrix.dot(self._simplebackdrop.w[tdx]).reshape(
            self.shape[1:]
        )

    def model_spline(self, tdx):
        """Returns the spline model at a given time index"""
        return self.spline_design_matrix.dot(self.spline_w[tdx]).reshape(
            (self.row.shape[0], self.column.shape[0])
        )

    def model_strap(self, tdx):
        """Returns the strap model at a given time index"""
        #        return self.strap_design_matrix.dot(self.strap_w[tdx]).reshape(self.shape[1:])
        return self.strap_w[0][(self.column * 2048 + self.bore_pixel[1]).astype(int)][
            None, :
        ] * np.ones((self.row.shape[0], self.column.shape[0]))

    def model(self, tdx):
        """Returns the full model at a given time index"""
        return self.model_simple(tdx) + self.model_spline(tdx) + self.model_strap(tdx)

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
        if not hasattr(self, "spline_w"):
            raise ValueError("Please run or load a model.")
        simple_bkg = self._simplebackdrop.build_model(
            column=column, row=row, times=times
        )

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

        self.column, self.row = (column - self.bore_pixel[1]) / (2048), (
            row - self.bore_pixel[0]
        ) / (2048)
        self._build_spline_design_matrix()
        bkg = np.zeros((len(tdxs), self.row.shape[0], self.column.shape[0]))
        for idx, tdx in enumerate(tdxs):
            bkg[idx, :, :] = self.model_spline(tdx) + self.model_strap(tdxs)
        return bkg + simple_bkg

    def plot(self, frame=0, vmin=None, vmax=None):
        """Plots a given frame of the data, model and residuals"""
        with plt.style.context("seaborn-white"):
            f = self.flux(frame)
            if not hasattr(self, "simple_design_matrix"):
                self._build_full_matrices()
            if np.nansum(f) == 0:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True, sharey=True)
                if vmin is None:
                    vmin = np.nanpercentile(self.model(frame), 1)
                if vmax is None:
                    vmax = np.nanpercentile(self.model(frame), 99)
                ax = [ax]
            else:
                fig, ax = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
                if vmin is None:
                    vmin = np.nanpercentile(f, 1)
                if vmax is None:
                    vmax = np.nanpercentile(f, 99)
            ax[0].imshow(self.model(frame), vmin=vmin, vmax=vmax, cmap="Greys_r")
            ax[0].set(xlabel="Column", ylabel="Row", title="Model")
            if np.nansum(f) == 0:
                return ax[0]
            ax[1].imshow(f, vmin=vmin, vmax=vmax, cmap="Greys_r")
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

    def _package_weights_hdulist(self):
        """Put the masks into a fits format"""
        cols = [
            fits.Column(
                name="column_knots_wbounds",
                format="D",
                unit="PIX",
                array=self.column_knots_wbounds,
            ),
            fits.Column(
                name="row_knots_wbounds",
                format="D",
                unit="PIX",
                array=self.row_knots_wbounds,
            ),
        ]
        hdu1 = fits.BinTableHDU.from_columns(cols, name="CART_KNOTS")
        hdu2 = fits.ImageHDU(self.spline_w, name="spline_weights")
        hdu3 = fits.ImageHDU(self.strap_w, name="strap_weights")
        hdul = fits.HDUList([hdu1, hdu2, hdu3])
        return hdul

    def save(self, output_dir=None, overwrite=False):
        """Save a model fit to the tess-backrop data directory.
        Will create a `simple` fits file containing the following extensions
        Will create a `spline` fits file containing the following extensions
        Will create a `straps` fits file containing the following extensions
        """
        if not hasattr(self, "spline_w"):
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
            "nknots",
            "degree",
            "njitter",
            "test_frame",
            "cutout_size",
        ]:
            self.hdu0.header[key] = getattr(self, key)
        self._simplebackdrop.hdu0 = self.hdu0
        if output_dir is None:
            output_dir = f"{PACKAGEDIR}/data/sector{self.sector:03}/camera{self.camera:02}/ccd{self.ccd:02}/"
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

        log.info(f"Saving s{self.sector} c{self.camera} ccd{self.ccd}")
        hdul = self._simplebackdrop._package_weights_hdulist()
        hdul[0] = self.hdu0
        _ = [hdul.append(i) for i in self._package_weights_hdulist()]
        fname = (
            f"tessbackdrop_sector{self.sector}_camera{self.camera}_ccd{self.ccd}.fits"
        )
        hdul.writeto(output_dir + fname, overwrite=overwrite)

    def load(self, input, nb=1, dir=None):
        """
        Load a model fit to the tess-backrop data directory.

        Parameters
        ----------
        input: tuple or string
            Either pass a tuple with `(sector, camera, ccd)` or pass
            a file name in `dir` to load
        """
        if isinstance(input, tuple):
            if len(input) == 3:
                sector, camera, ccd = input
                fname = f"tessbackdrop_sector{sector}_camera{camera}_ccd{ccd}.fits"
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
        self._simplebackdrop.load(fname, nb=1, dir=dir)
        with fits.open(dir + fname, lazy_load_hdus=True) as hdu:
            for key in [
                "nknots",
                "degree",
            ]:
                setattr(self, key, hdu[0].header[key])
            self.spline_w = hdu[6].data
            self.strap_w = hdu[7].data
        self.__post_init__()
        self._simplebackdrop.load(fname, nb=1, dir=dir)

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

        if np.any([not hasattr(self, attr) for attr in ["camera", "ccd", "sector"]]):
            self.load((tpf.sector, tpf.camera, tpf.ccd))
        elif (
            (self.sector != tpf.sector)
            | (self.camera != tpf.camera)
            | (self.ccd != tpf.ccd)
        ):
            self.load((tpf.sector, tpf.camera, tpf.ccd))

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
