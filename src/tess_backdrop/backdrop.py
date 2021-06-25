import logging
import os

import fitsio
import numpy as np
import pandas as pd
from astropy.io import fits
from lightkurve.correctors.designmatrix import _spline_basis_vector
from scipy.sparse import csr_matrix, hstack, lil_matrix, vstack

from . import PACKAGEDIR
from .version import __version__

log = logging.getLogger(__name__)


class BackDrop(object):
    """
    Class to create background corrections for TESS.

    `tess-backdrop` fits a simple, three part model:
    1. A 2D, low order polynomial to remove the bulk background
    2. A 2D b-spline to remove high spatial frequency noise
    3. A model for strap offsets.

    Parameters
    ----------
    fnames: list of str, or list of astropy.io.fits objects
        Input TESS FFIs
    npoly: int
        The order of polynomial to fit to the data in both x and y dimension.
        Recommend ~4.
    nrad: int
        The order of polynomial to fit to the data in radius from the boresight
    nknots: int
        Number of knots to fit in each dimension. Recommend ~40.
    degree: int
        Degree of spline to fit.
    nb: int
        Number of bins to downsample to for polynomial fit.
    cutout_size : int
        Size of cut out to use. Default is 2048, full FFI
    """

    def __init__(
        self,
        fnames=None,
        npoly=5,
        nrad=5,
        nknots=40,
        degree=3,
        nb=8,
        cutout_size=2048,
        max_batch_number=20,
        min_batch_size=5,
        #        reference_frame=0,
    ):

        """
        Parameters
        ----------
        fnames: list of str, or list of astropy.io.fits objects
            Input TESS FFIs
        npoly: int
            The order of polynomial to fit to the data in both x and y dimension.
            Recommend ~4.
        nknots: int
            Number of knots to fit in each dimension. Recommend ~40.
        degree: int
            Degree of spline to fit.
        nb: int
            Number of bins to downsample to for polynomial fit.
        """

        self.npoly = npoly
        self.nrad = nrad
        self.nknots = nknots
        self.degree = degree
        self.fnames = fnames
        self.nknots = nknots
        self.max_batch_number = max_batch_number
        self.min_batch_size = min_batch_size
        self.nb = nb
        self.cutout_size = cutout_size
        #        self.reference_frame = reference_frame
        #        if self.fnames is not None:
        #            self.reference_image = self.fnames[self.reference_frame]
        if self.fnames is not None:
            if isinstance(self.fnames, (str, list)):
                self.fnames = np.asarray(self.fnames)
            self.fnames = np.sort(self.fnames)
            if len(self.fnames) >= 15:
                log.info("Finding bad frames")
                self.bad_frames, self.quality = _find_bad_frames(
                    self.fnames, cutout_size=self.cutout_size
                )
            else:
                self.bad_frames = np.zeros(len(self.fnames), bool)

            if (
                len(self.fnames[~self.bad_frames]) / self.max_batch_number
                < self.min_batch_size
            ):
                self.batches = np.array_split(
                    self.fnames[~self.bad_frames],
                    np.max(
                        [1, len(self.fnames[~self.bad_frames]) // self.min_batch_size]
                    ),
                )
            else:
                self.batches = np.array_split(
                    self.fnames[~self.bad_frames], self.max_batch_number
                )

        else:
            self.bad_frames = None
        self.knots_wbounds = _get_knots(
            np.arange(self.cutout_size), nknots=nknots, degree=degree
        )

    def _build_mask(self):
        """Builds a boolean mask for the input image stack which

        1. Masks out very bright pixels (>1500 counts) or pixels with a sharp gradient (>30 counts)
        2. Masks out pixels where there is a consistently outlying gradient in the image
        3. Masks out saturated columns, including a wide halo, and a whisker mask.

        Returns
        -------
        soft_mask : np.ndarray of bools
            "soft" mask of pixels that, on average, have steep gradients
        sat_mask : np.ndarray of bools
            Mask where pixels that are not saturated are True
        """
        #        average = np.zeros((self.cutout_size, self.cutout_size))
        #        weights = np.zeros((self.cutout_size, self.cutout_size))
        diff = None
        # diff_ar = np.zeros(len(self.fnames))
        # sat_mask = None
        # hard_mask = np.zeros((self.cutout_size, self.cutout_size), dtype=bool)

        med_image = np.zeros((len(self.batches), self.cutout_size, self.cutout_size))
        self.odd_mask = csr_matrix((len(self.fnames), 2048 ** 2), dtype=bool)

        for bdx, batch in enumerate(self.batches):
            if len(batch) == 0:
                med_image[bdx, :, :] = np.nan
            else:
                batch_count = 0
                for fdx, fname in enumerate(batch):
                    fdx += np.where(self.fnames == batch[0])[0][0]
                    with fits.open(fname, lazy_load_hdus=True) as hdu:
                        if (fname == self.fnames[0]) & (fdx == 0):
                            self.sector = int(fname.split("-s")[1].split("-")[0])
                            self.camera = hdu[1].header["camera"]
                            self.ccd = hdu[1].header["ccd"]
                            if self.ccd in [1, 3]:
                                self.bore_pixel = [2048, 2048]
                            elif self.ccd in [2, 4]:
                                self.bore_pixel = [2048, 0]
                            log.info(
                                f"Building mask s{self.sector} c{self.camera} ccd{self.ccd}"
                            )
                        if not np.all(
                            [
                                self.sector == int(fname.split("-s")[1].split("-")[0]),
                                self.camera == hdu[1].header["camera"],
                                self.ccd == hdu[1].header["ccd"],
                            ]
                        ):
                            raise ValueError(
                                "All files must have same sector, camera, ccd."
                            )
                    # Bad frames do not count.
                    if self.bad_frames[fdx]:
                        continue
                    data = fitsio.read(fname)[
                        : self.cutout_size, 45 : self.cutout_size + 45
                    ]
                    k = (data > 0) & (data < 1500)
                    # Blown out frames do not count.
                    if (~k).sum() / (self.cutout_size ** 2) > 0.05:
                        self.bad_frames[fdx] = True
                        continue
                    data -= np.median(data[::16, ::16])
                    if diff is None:
                        diff = data.copy()
                    else:
                        diff -= data
                        # diff_ar[fdx] = (np.abs(diff) > 200).sum() / (
                        #    self.cutout_size ** 2
                        # )
                        if (np.abs(diff) > 200).sum() / (self.cutout_size ** 2) > 0.2:
                            diff = data.copy()
                            self.bad_frames[fdx] = True
                            continue
                        diff = data.copy()

                    med_image[bdx, :, :] += data
                    batch_count += 1

                    # grad = np.gradient(data)
                    # hard_mask |= np.abs(np.hypot(*grad)) > 50
                if batch_count > 0:
                    med_image[bdx, :, :] /= batch_count
                else:
                    med_image[bdx, :, :] = np.nan

        med_image = np.nanmedian(med_image, axis=0)
        self.average_image = med_image
        del med_image
        self.average_image -= np.nanmedian(self.average_image)
        self._straps_removed_from_average = False

        soft_mask = np.hypot(*np.gradient(self.average_image)) > 10  # | (weights == 0)

        sat_mask = get_saturation_mask(self.average_image, cutout_size=self.cutout_size)
        # asteroid_mask = np.zeros((self.cutout_size, self.cutout_size), dtype=bool)
        #
        # binsize = 128
        # for fdx, fname in enumerate(self.fnames):
        #     data = fitsio.read(fname)[: self.cutout_size, 45 : self.cutout_size + 45]
        #     data -= np.median(data)
        #     data -= self.average_image
        #     m = (data > 500) | (data < 0) & soft_mask
        #     if self.cutout_size > binsize:
        #         check = np.asarray(
        #             [
        #                 m[idx::binsize, jdx::binsize]
        #                 for idx in range(binsize)
        #                 for jdx in range(binsize)
        #             ]
        #         ).sum(axis=0) / (binsize ** 2)
        #     else:
        #         check = m
        #     if (check > 0.05).any():
        #         self.odd_mask[fdx] = csr_matrix(m.ravel())
        #     check = np.kron(
        #         check,
        #         np.ones(
        #             (binsize, binsize),
        #             dtype=int,
        #         ),
        #     )
        #     grad = np.gradient(np.abs(data))
        #     asteroid_mask |= (np.hypot(*grad) > 30) & (check < 0.05)
        #
        # import pdb
        # import matplotlib.pyplot as plt
        #
        # pdb.set_trace()
        # del data

        # # I don't care about dividing by zero here
        # with np.errstate(divide="ignore", invalid="ignore"):
        #     average /= weights

        # soft_mask = (np.hypot(*np.gradient(average)) > 10) | (weights == 0)
        # soft_mask = (average > 20) | (weights == 0)
        # soft_mask |= np.any(np.gradient(soft_mask.astype(float)), axis=0) != 0

        # This makes the soft mask slightly more generous
        def enlarge_mask(mask):
            m = np.zeros((self.cutout_size, self.cutout_size))
            m[1:-1, 1:-1] += mask[:-2, 1:-1].astype(int)
            m[1:-1, 1:-1] += mask[2:, 1:-1].astype(int)
            m[1:-1, 1:-1] += mask[1:-1, :-2].astype(int)
            m[1:-1, 1:-1] += mask[1:-1, 2:].astype(int)
            mask |= m >= 3

        enlarge_mask(soft_mask)
        #        enlarge_mask(asteroid_mask)

        #        self.star_mask = ~(hard_mask | asteroid_mask | soft_mask | ~sat_mask)
        self.star_mask = ~(soft_mask | ~sat_mask)
        self.sat_mask = sat_mask

        # We don't need all these pixels, it's too many to store for every frame.
        # Instead we'll just save 5000 of them.
        if (soft_mask & sat_mask).sum() > 5000:
            s = np.random.choice((soft_mask & sat_mask).sum(), size=5000, replace=False)
            l = np.asarray(np.where(soft_mask & sat_mask))
            l = l[:, s]
            self.jitter_mask = np.zeros((self.cutout_size, self.cutout_size), bool)
            self.jitter_mask[l[0], l[1]] = True
        else:
            self.jitter_mask = np.copy((soft_mask & sat_mask))
        # fname = self.fnames[len(self.fnames) // 2]
        # data = fitsio.read(fname)[: self.cutout_size, 45 : self.cutout_size + 45]
        # grad = np.asarray(np.gradient(data))
        # self.median_data = data[self.jitter_mask]
        # self.median_gradient = grad[:, self.jitter_mask]
        self.median_data = self.average_image[self.jitter_mask]
        self.median_gradient = np.asarray(np.gradient(self.average_image))[
            :, self.jitter_mask
        ]

        return soft_mask, sat_mask  # , diff_ar

    def _build_matrices(self):
        """Allocate the matrices to fit the background.
        When we want to build the matrices to evaluate this background model,
        we will be able do so in a slightly more efficient way."""
        log.info(f"Building matrices s{self.sector} c{self.camera} ccd{self.ccd}")

        row, column = np.mgrid[: self.cutout_size, : self.cutout_size]
        c, r = column / self.cutout_size - 0.5, row / self.cutout_size - 0.5
        crav = c.ravel()
        rrav = r.ravel()

        self._poly_X = np.asarray(
            [
                crav ** idx * rrav ** jdx
                for idx in np.arange(self.npoly)
                for jdx in np.arange(self.npoly)
            ]
        ).T

        c, r = (column - self.bore_pixel[1]) / 2048, (row - self.bore_pixel[0]) / 2048
        crav = c.ravel()
        rrav = r.ravel()
        rad = (crav ** 2 + rrav ** 2)[:, None] ** 0.5

        self._poly_X = np.hstack(
            [self._poly_X, np.hstack([rad ** idx for idx in np.arange(1, self.nrad)])]
        )

        def expand_poly(x, crav, points):
            points = np.arange(0, 2048 + 512, 512)
            return np.hstack(
                [
                    x
                    * (
                        (((crav + 0.5) * self.cutout_size) >= p1)
                        & (((crav + 0.5) * self.cutout_size) < p2)
                    )[:, None]
                    for p1, p2 in zip(points[:-1], points[1:])
                    if (
                        (((crav + 0.5) * self.cutout_size) >= p1)
                        & (((crav + 0.5) * self.cutout_size) < p2)
                    ).any()
                ]
            )

        # self._poly_X = expand_poly(self._poly_X, crav, points)
        del (
            row,
            column,
            c,
            r,
        )
        del crav, rrav

        row, column = (
            np.mgrid[: self.cutout_size // self.nb, : self.cutout_size // self.nb]
            * self.nb
        )
        row, column = row + self.nb / 2, column + self.nb / 2
        c, r = column / self.cutout_size - 0.5, row / self.cutout_size - 0.5
        crav = c.ravel()
        rrav = r.ravel()

        self.weights = np.sum(
            [
                self.star_mask[idx :: self.nb, jdx :: self.nb]
                for idx in range(self.nb)
                for jdx in range(self.nb)
            ],
            axis=0,
        )

        self._poly_X_down = np.asarray(
            [
                crav ** idx * rrav ** jdx
                for idx in np.arange(self.npoly)
                for jdx in np.arange(self.npoly)
            ]
        ).T

        c, r = (column - self.bore_pixel[1]) / 2048, (row - self.bore_pixel[0]) / 2048
        crav = c.ravel()
        rrav = r.ravel()
        rad = (crav ** 2 + rrav ** 2)[:, None] ** 0.5
        self._poly_X_down = np.hstack(
            [
                self._poly_X_down,
                np.hstack([rad ** idx for idx in np.arange(1, self.nrad)]),
            ]
        )

        # self._poly_X_down = expand_poly(self._poly_X_down, crav, points)
        del (
            row,
            column,
            c,
            r,
        )
        del crav, rrav

        self.poly_sigma_w_inv = self._poly_X_down[self.weights.ravel() != 0].T.dot(
            self._poly_X_down[self.weights.ravel() != 0]
        )

        e = lil_matrix((self.cutout_size, self.cutout_size * self.cutout_size))
        for idx in range(self.cutout_size):
            e[idx, np.arange(self.cutout_size) * self.cutout_size + idx] = 1

        self._strap_X = e.T.tocsr()
        del e
        self._spline_X = self._get_spline_matrix(np.arange(self.cutout_size))
        self.X = hstack([self._spline_X, self._strap_X], format="csr")

        # We'll sacrifice some memory here for speed later.
        # self.XT = self.X.T.tocsr()
        # self.Xm = self.X[self.star_mask.ravel()].tocsr()
        self.XmT = self.X[self.star_mask.ravel()].T.tocsr()
        self.prior_mu = np.zeros(self._spline_X.shape[1] + self._strap_X.shape[1])
        self.prior_sigma = (
            np.ones(self._spline_X.shape[1] + self._strap_X.shape[1]) * 40
        )
        self.prior_sigma[: self._spline_X.shape[1]] *= 10
        self.sigma_w_inv = self.XmT.dot(
            self.X[self.star_mask.ravel()].tocsr()
        ) + np.diag(1 / self.prior_sigma ** 2)
        if not self._straps_removed_from_average:
            log.info(
                f"Correcting average image s{self.sector} c{self.camera} ccd{self.ccd}"
            )
            self._straps_removed_from_average = True
            fit_results = self._fit_frame(self.average_image)
            self.average_image -= fit_results[2][None, :]
            self.average_image -= self._poly_X.dot(fit_results[0]).reshape(
                (self.cutout_size, self.cutout_size)
            )
            self.average_image -= self._spline_X.dot(fit_results[1].ravel()).reshape(
                (self.cutout_size, self.cutout_size)
            )

    def _get_spline_matrix(self, xc, xr=None):
        """Helper function to make a 2D spline matrix in a fairly memory efficient way."""

        def _X(x):
            matrices = [
                csr_matrix(
                    _spline_basis_vector(x, self.degree, idx, self.knots_wbounds)
                )
                for idx in np.arange(-1, len(self.knots_wbounds) - self.degree - 1)
            ]
            X = vstack(matrices, format="csr").T
            return X

        if xr is None:
            xr = xc
        Xc = _X(xc)
        Xcf = vstack([Xc for idx in range(len(xr))]).tocsr()
        Xr = _X(xr)
        Xrf = (
            hstack([Xr for idx in range(len(xc))])
            .reshape((Xcf.shape[0], Xc.shape[1]))
            .tocsr()
        )
        Xf = hstack([Xrf.multiply(X.T) for X in Xcf.T]).tocsr()
        return Xf

    def __repr__(self):
        if hasattr(self, "sector"):
            return (
                f"BackDrop [Sector {self.sector}, Camera {self.camera}, CCD {self.ccd}]"
            )
        return "BackDrop"

    def fit_model(self):
        """Fit the tess-backdrop model to the files specified by `fnames`."""
        if not hasattr(self, "star_mask"):
            _ = self._build_mask()
        if not hasattr(self, "_poly_X"):
            self._build_matrices()

        if not hasattr(self, "poly_w"):
            self.poly_w, self.spline_w, self.strap_w, self.t_start, self.jitter = (
                # np.zeros(
                #     (
                #         len(self.fnames),
                #         self.npoly
                #         * self.npoly
                #         * (np.arange(0, 2048 + 512, 512) < self.cutout_size).sum(),
                #     )
                # ),
                # np.zeros((len(self.fnames), self.npoly, self.npoly)),
                np.zeros((len(self.fnames), self.npoly * self.npoly + self.nrad - 1)),
                np.zeros((len(self.fnames), self.nknots, self.nknots)),
                np.zeros((len(self.fnames), self.cutout_size)),
                np.zeros(len(self.fnames)),
                np.zeros((len(self.fnames), self.jitter_mask.sum())),
            )
        log.info(f"Building frames s{self.sector} c{self.camera} ccd{self.ccd}")
        points = np.linspace(0, len(self.fnames), 12, dtype=int)

        for idx, fname in enumerate(self.fnames):
            if self.t_start[idx] != 0:
                continue
            if idx in points:
                log.info(
                    f"Running frames s{self.sector} c{self.camera} ccd{self.ccd} {np.where(points == idx)[0][0] * 10}%"
                )
                if idx != 0:
                    self.save()
            with fits.open(fname, lazy_load_hdus=True) as hdu:
                if not np.all(
                    [
                        self.sector == int(fname.split("-s")[1].split("-")[0]),
                        self.camera == hdu[1].header["camera"],
                        self.ccd == hdu[1].header["ccd"],
                    ]
                ):
                    raise ValueError(
                        f"FFI image is not part of Sector {self.sector}, Camera {self.camera}, CCD {self.ccd}"
                    )
                self.t_start[idx] = hdu[0].header["TSTART"]
            data = (
                fitsio.read(fname)[: self.cutout_size, 45 : self.cutout_size + 45]
                - self.average_image
            )
            (
                self.poly_w[idx, :],
                self.spline_w[idx, :],
                self.strap_w[idx, :],
                self.jitter[idx, :],
            ) = self._fit_frame(data)
        self.save()
        # # Smaller version of jitter for use later
        # bad = sigma_clip(np.gradient(self.jitter_pix, axis=1).std(axis=1), sigma=5).mask
        # _, med, std = sigma_clipped_stats(self.jitter_pix[~bad], axis=0)
        # j = (np.copy(self.jitter_pix) - med) / std
        # j[j > 10] = 0
        # U, s, V = pca(j[~bad], 20, n_iter=100)
        # X = np.zeros((self.jitter_pix.shape[0], U.shape[1]))
        # X[~bad] = np.copy(U)
        # self.jitter = X
        # self.jitter = np.copy(self.jitter_pix)

    def _fit_frame(self, data):
        """Helper function to fit a model to an individual frame."""

        avg = np.sum(
            [
                data[idx :: self.nb, jdx :: self.nb]
                * self.star_mask[idx :: self.nb, jdx :: self.nb]
                for idx in range(self.nb)
                for jdx in range(self.nb)
            ],
            axis=0,
        )

        # I don't care about dividing by zero here
        with np.errstate(divide="ignore", invalid="ignore"):
            avg /= self.weights
        B = self._poly_X_down[self.weights.ravel() != 0].T.dot(
            avg.ravel()[self.weights.ravel() != 0]
        )
        poly_w = np.linalg.solve(self.poly_sigma_w_inv, B)

        # iterate once
        for count in [0, 1]:
            m = self._poly_X_down.dot(poly_w).reshape(
                (self.cutout_size // self.nb, self.cutout_size // self.nb)
            )

            k = np.abs(avg - m) < 300
            k = (self.weights.ravel() != 0) & k.ravel()
            poly_sigma_w_inv = self._poly_X_down[k].T.dot(self._poly_X_down[k])
            B = self._poly_X_down[k].T.dot(avg.ravel()[k])
            poly_w = np.linalg.solve(poly_sigma_w_inv, B)

        res = data - self._poly_X.dot(poly_w).reshape(
            (self.cutout_size, self.cutout_size)
        )
        res[(np.hypot(*np.gradient(np.abs(res))) > 30) | (np.abs(res) > 500)] *= 0

        # The spline and strap components should be small
        # sigma_w_inv = self.XT[:, star_mask.ravel()].dot(
        #        self.X[star_mask.ravel()]
        #    ).toarray() + np.diag(1 / prior_sigma ** 2)
        B = (
            self.XmT.dot(res.ravel()[self.star_mask.ravel()])
            + self.prior_mu / self.prior_sigma ** 2
        )
        w = np.linalg.solve(self.sigma_w_inv, B)

        spline_w = w[: self._spline_X.shape[1]].reshape((self.nknots, self.nknots))
        strap_w = w[self._spline_X.shape[1] :]
        jitter_pix = res[self.jitter_mask]

        return (
            poly_w,  # .reshape((self.npoly, self.npoly)),
            spline_w,
            strap_w,
            jitter_pix,
        )

    def _get_jitter(self, data):
        """Get the jitter correction somehow..."""
        # Use jitter mask.

        raise NotImplementedError

    def save(self, output=None):
        """
        Save a model fit to the tess-backrop data directory.

        Will create a fits file containing the following extensions
            - Primary
            - T_START: The time array for each background solution
            - KNOTS: Knot spacing in row and column
            - SPLINE_W: Solution to the spline model. Has shape (ntimes x nknots x nknots)
            - STRAP_W: Solution to the strap model. Has shape (ntimes x self.cutout_size)
            - POLY_W: Solution to the polynomial model. Has shape (ntimes x npoly x npoly)
        """
        log.info(f"Saving s{self.sector} c{self.camera} ccd{self.ccd}")
        # if not hasattr(self, "star_mask"):
        #     raise ValueError(
        #         "It does not look like you have regenerated a tess_backdrop model, I do not think you want to save."
        #     )
        hdu0 = fits.PrimaryHDU()
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
        hdu1 = fits.BinTableHDU.from_columns(cols)
        cols = [
            fits.Column(name="KNOTS", format="D", unit="PIX", array=self.knots_wbounds)
        ]
        hdu2 = fits.BinTableHDU.from_columns(cols)
        hdu3 = fits.ImageHDU(self.spline_w[s], name="spline_w")
        hdu4 = fits.ImageHDU(self.strap_w[s], name="strap_w")
        hdu5 = fits.ImageHDU(self.poly_w[s], name="poly_w")
        hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4, hdu5])
        hdul[0].header["ORIGIN"] = "tess-backdrop"
        hdul[0].header["AUTHOR"] = "christina.l.hedges@nasa.gov"
        hdul[0].header["VERSION"] = __version__

        for key in ["sector", "camera", "ccd", "nknots", "npoly", "nrad", "degree"]:
            hdul[0].header[key] = getattr(self, key)

        fname = (
            f"tessbackdrop_sector{self.sector}_camera{self.camera}_ccd{self.ccd}.fits"
        )
        dir = f"{PACKAGEDIR}/data/sector{self.sector:03}/camera{self.camera:02}/ccd{self.ccd:02}/"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        hdul.writeto(dir + fname, overwrite=True)

        hdu0 = fits.PrimaryHDU()
        hdu1 = fits.ImageHDU(self.jitter[s], name="jitter_pix")
        hdul = fits.HDUList([hdu0, hdu1])
        hdul[0].header["ORIGIN"] = "tess-backdrop"
        hdul[0].header["AUTHOR"] = "christina.l.hedges@nasa.gov"
        hdul[0].header["VERSION"] = __version__
        for key in ["sector", "camera", "ccd", "nknots", "npoly", "nrad", "degree"]:
            hdul[0].header[key] = getattr(self, key)
        if output is None:
            fname = f"tessbackdrop_jitter_sector{self.sector}_camera{self.camera}_ccd{self.ccd}.fits"
            dir = f"{PACKAGEDIR}/data/sector{self.sector:03}/camera{self.camera:02}/ccd{self.ccd:02}/"
            hdul.writeto(dir + fname, overwrite=True)
        else:
            hdul.writeto(output, overwrite=True)

    def load(self, sector, camera, ccd):
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
        fname = f"tessbackdrop_sector{sector}_camera{camera}_ccd{ccd}.fits"
        with fits.open(dir + fname, lazy_load_hdus=True) as hdu:
            for key in ["sector", "camera", "ccd", "nknots", "npoly", "nrad", "degree"]:
                setattr(self, key, hdu[0].header[key])
            self.t_start = hdu[1].data["T_START"]
            if "QUALITY" in hdu[1].data.names:
                self.quality = hdu[1].data["QUALITY"]
            self.knots_wbounds = hdu[2].data["KNOTS"]
            self.spline_w = hdu[3].data
            self.strap_w = hdu[4].data
            self.poly_w = hdu[5].data

        fname = f"tessbackdrop_jitter_sector{sector}_camera{camera}_ccd{ccd}.fits"
        with fits.open(dir + fname, lazy_load_hdus=True) as hdu:
            self.jitter = hdu[1].data
        if self.ccd in [1, 3]:
            self.bore_pixel = [2048, 2048]
        elif self.ccd in [2, 4]:
            self.bore_pixel = [2048, 0]

    def list_available(self):
        """List the sectors, cameras and CCDs that
        are available to you via the `load` method.

        If there is a sector that is not available that you need,
        you can create a solution using the TESS FFIs, and then use the
        `save` method."""

        df = pd.DataFrame(columns=["Sector", "Camera", "CCD"])
        idx = 0
        for sector in np.arange(200):
            for camera in np.arange(1, 5):
                for ccd in np.arange(1, 5):
                    dir = f"{PACKAGEDIR}/data/sector{sector:03}/camera{camera:02}/ccd{ccd:02}/"
                    if not os.path.isdir(dir):
                        continue
                    fname = f"tessbackdrop_sector{sector}_camera{camera}_ccd{ccd}.fits"
                    if os.path.isfile(dir + fname):
                        df.loc[idx] = np.hstack([sector, camera, ccd])
                        idx += 1
        return df

    def build_correction(self, column, row, times=None):
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
            raise ValueError(
                "tess-backdrop does not have any backdrop information. Do you need to `load` a backdrop file?"
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

        c, r = np.meshgrid(column, row)
        c, r = c / self.cutout_size - 0.5, r / self.cutout_size - 0.5
        self._poly_X = np.asarray(
            [
                c.ravel() ** idx * r.ravel() ** jdx
                for idx in np.arange(self.npoly)
                for jdx in np.arange(self.npoly)
            ]
        ).T

        c, r = np.meshgrid(column, row)
        c, r = (c - self.bore_pixel[1]) / 2048, (r - self.bore_pixel[1]) / 2048
        crav = c.ravel()
        rrav = r.ravel()
        rad = (crav ** 2 + rrav ** 2)[:, None] ** 0.5

        self._poly_X = np.hstack(
            [self._poly_X, np.hstack([rad ** idx for idx in np.arange(1, self.nrad)])]
        )

        del c, r, crav, rrav
        self._spline_X = self._get_spline_matrix(column, row)
        bkg = np.zeros((len(tdxs), len(row), len(column)))
        for idx, tdx in enumerate(tdxs):
            poly = self._poly_X.dot(self.poly_w[tdx].ravel()).reshape(
                (row.shape[0], column.shape[0])
            )
            spline = self._spline_X.dot(self.spline_w[tdx].ravel()).reshape(
                (row.shape[0], column.shape[0])
            )
            strap = self.strap_w[tdx][column][None, :] * np.ones(row.shape[0])[:, None]
            bkg[idx, :, :] = poly + spline + strap
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

        bkg = self.build_correction(
            np.arange(tpf.shape[2]) + tpf.column,
            np.arange(tpf.shape[1]) + tpf.row,
            times=tdxs,
        )
        return tpf - bkg


def _get_knots(x, nknots, degree):
    """Find the b-spline knot spacing for an input array x, number of knots and degree

    Parameters
    ----------
    x : np.ndarray
        In put vector to create b-spline for
    nknots : int
        Number of knots to use in the b-spline
    degree : int
        Degree of the b-spline

    Returns
    -------
    knots_wbounds : np.ndarray
        The knot locations for the input x.
    """

    knots = np.asarray(
        [s[-1] for s in np.array_split(np.argsort(x), nknots - degree)[:-1]]
    )
    knots = [np.mean([x[k], x[k + 1]]) for k in knots]
    knots = np.append(np.append(x.min(), knots), x.max())
    knots = np.unique(knots)
    knots_wbounds = np.append(
        np.append([x.min()] * (degree - 1), knots), [x.max()] * (degree)
    )
    return knots_wbounds


def _find_saturation_column_centers(mask):
    """
    Finds the center point of saturation columns.

    Parameters
    ----------
    mask : np.ndarray of bools
        Mask where True indicates a pixel is saturated

    Returns
    -------
    centers : np.ndarray
        Array of the centers in XY space for all the bleed columns
    """
    centers = []
    radii = []
    idxs = np.where(mask.any(axis=0))[0]
    for idx in idxs:
        line = mask[:, idx]
        seq = []
        val = line[0]
        jdx = 0

        while jdx <= len(line):
            while line[jdx] == val:
                jdx += 1
                if jdx >= len(line):
                    break

            if jdx >= len(line):
                break
            seq.append(jdx)
            val = line[jdx]
        w = np.array_split(line, seq)
        v = np.array_split(np.arange(len(line)), seq)
        coords = [(idx, v1.mean().astype(int)) for v1, w1 in zip(v, w) if w1.all()]
        rads = [len(v1) / 2 for v1, w1 in zip(v, w) if w1.all()]
        for coord, rad in zip(coords, rads):
            centers.append(coord)
            radii.append(rad)
    centers = np.asarray(centers)
    radii = np.asarray(radii)
    return centers, radii


def get_saturation_mask(data, whisker_width=40, cutout_size=2048):
    """
    Finds a mask that will remove saturated pixels, and any "whiskers".

    Parameters
    ----------
    data : np.ndarray of shape (2048 x 2048)
        Input TESS FFI

    Returns
    -------
    sat_mask: np.ndarray of bools
        The mask for saturated pixels. False where pixels are saturated.
    """
    sat_cols = (np.abs(np.gradient(data)[1]) > 1e4) | (data > 1e5)

    centers, radii = _find_saturation_column_centers(sat_cols)
    whisker_mask = np.zeros((cutout_size, cutout_size), bool)
    for idx in np.arange(-2, 2):
        for jdx in np.arange(-whisker_width // 2, whisker_width // 2):

            a1 = np.max([np.zeros(len(centers)), centers[:, 1] - idx], axis=0)
            a1 = np.min([np.ones(len(centers)) * cutout_size - 1, a1], axis=0)

            b1 = np.max([np.zeros(len(centers)), centers[:, 0] - jdx], axis=0)
            b1 = np.min([np.ones(len(centers)) * cutout_size - 1, b1], axis=0)

            whisker_mask[a1.astype(int), b1.astype(int)] = True

    sat_mask = np.copy(sat_cols)
    sat_mask |= np.gradient(sat_mask.astype(float), axis=0) != 0
    for count in range(4):
        sat_mask |= np.gradient(sat_mask.astype(float), axis=1) != 0
    sat_mask |= whisker_mask

    X, Y = np.mgrid[:cutout_size, :cutout_size]

    jdx = 0
    kdx = 0
    for jdx in range(8):
        for kdx in range(8):
            k = (
                (centers[:, 1] > jdx * 256 - radii.max() - 1)
                & (centers[:, 1] <= (jdx + 1) * 256 + radii.max() + 1)
                & (centers[:, 0] > kdx * 256 - radii.max() - 1)
                & (centers[:, 0] <= (kdx + 1) * 256 + radii.max() + 1)
            )
            if not (k).any():
                continue
            for idx in np.where(k)[0]:
                x, y = (
                    X[jdx * 256 : (jdx + 1) * 256, kdx * 256 : (kdx + 1) * 256]
                    - centers[idx][1],
                    Y[jdx * 256 : (jdx + 1) * 256, kdx * 256 : (kdx + 1) * 256]
                    - centers[idx][0],
                )
                sat_mask[
                    jdx * 256 : (jdx + 1) * 256, kdx * 256 : (kdx + 1) * 256
                ] |= np.hypot(x, y) < (np.min([radii[idx], 70]))

    # for idx in tqdm(range(len(centers)), desc="Building Saturation Mask"):
    #     sat_mask |= np.hypot(X - centers[idx][1], Y - centers[idx][0]) < (radii[idx])
    return ~sat_mask


def _std_iter(x, mask, sigma=3, n_iters=3):
    """Iteratively finds the standard deviation of an array after sigma clipping
    Parameters
    ----------
    x : np.ndarray
        Array with average of zero
    mask : np.ndarray of bool
        Mask of same size as x, where True indicates a point to be masked.
    sigma : int or float
        The standard deviation at which to clip
    n_iters : int
        Number of iterations
    """
    m = mask.copy()
    for iter in range(n_iters):
        std = np.std(x[~m])
        m |= np.abs(x) > (std * sigma)
    return std


def _find_bad_frames(fnames, cutout_size=2048, corner_check=False):
    """Identifies frames that probably have a lot of scattered lightkurve
    If quality flags are available, will use TESS quality flags.

    If unavailable, or if `corner_check=True`, loads the 30x30 pixel corner
    region of every frame, and uses them to find frames that have a lot of
    scattered light.


    """

    quality = np.zeros(len(fnames), int)
    warned = False
    for idx, fname in enumerate(fnames):
        try:
            quality[idx] = fitsio.read_header(fname, 1)["DQUALITY"]
        except KeyError:
            if warned is False:
                log.warning("Quality flags are missing.")
                warned = True
            continue
    bad = (quality & (2048 | 175)) != 0

    if warned | corner_check:
        corner = np.zeros((4, len(fnames)))
        for tdx, fname in enumerate(fnames):
            corner[0, tdx] = fitsio.read(fname)[:30, 45 : 45 + 30].mean()
            corner[1, tdx] = fitsio.read(fname)[-30:, 45 : 45 + 30].mean()
            corner[2, tdx] = fitsio.read(fname)[
                :30, 45 + cutout_size - 30 : 45 + cutout_size
            ].mean()
            corner[3, tdx] = fitsio.read(fname)[
                -30:, 45 + cutout_size - 30 - 1 : 45 + cutout_size
            ].mean()

        c = corner.T - np.median(corner, axis=1)
        c /= np.std(c, axis=0)
        bad = (np.abs(c) > 2).any(axis=1)
        #    bad |= corner.std(axis=0) > 200
    return bad, quality
