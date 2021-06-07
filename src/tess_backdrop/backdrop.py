from astropy.io import fits
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from fbpca import pca

from scipy.sparse import csr_matrix, vstack, hstack, lil_matrix
from lightkurve.correctors.designmatrix import _spline_basis_vector
import fitsio

from astropy.stats import sigma_clip, sigma_clipped_stats
from . import PACKAGEDIR
from .version import __version__


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
    nknots: int
        Number of knots to fit in each dimension. Recommend ~40.
    degree: int
        Degree of spline to fit.
    nb: int
        Number of bins to downsample to for polynomial fit.
    """

    def __init__(
        self,
        fnames=None,
        npoly=4,
        nknots=40,
        degree=3,
        nb=8,
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
        self.nknots = nknots
        self.degree = degree
        self.fnames = fnames
        self.nknots = nknots
        self.nb = nb
        #        self.reference_frame = reference_frame
        #        if self.fnames is not None:
        #            self.reference_image = self.fnames[self.reference_frame]
        self.knots_wbounds = _get_knots(np.arange(2048), nknots=nknots, degree=degree)

    def _build_mask(self):
        """Builds a boolean mask for the input image stack which

        1. Masks out very bright pixels (>1500 counts) or pixels with a sharp gradient (>30 counts)
        2. Masks out pixels where there is a consistently outlying gradient in the image
        3. Masks out saturated columns, including a wide halo, and a whisker mask.

        Returns
        -------
        soft_mask : np.ndarray of bools
            "soft" mask of pixels that, on average, have steep gradients
        hard_mask : np.ndarray of bools
            "hard" mask of pixels that, in any frame, have a gradient of over 60 counts
        sat_mask : np.ndarray of bools
            Mask where pixels that are not saturated are True
        diff_ar : np.ndarray of bools
            Array with the fraction of pixels in a given frame that are 200 counts
            different from the previous frame.
        """
        average = np.zeros((2048, 2048))
        weights = np.zeros((2048, 2048))
        diff = None
        diff_ar = np.zeros(len(self.fnames))
        sat_mask = None
        hard_mask = np.zeros((2048, 2048), dtype=bool)
        for fdx, fname in enumerate(tqdm(self.fnames, desc="Building Pixel Mask")):
            with fits.open(fname, lazy_load_hdus=True) as hdu:
                if fname == self.fnames[0]:
                    self.sector = int(fname.split("-s")[1].split("-")[0])
                    self.camera = hdu[1].header["camera"]
                    self.ccd = hdu[1].header["ccd"]
                if not np.all(
                    [
                        self.sector == int(fname.split("-s")[1].split("-")[0]),
                        self.camera == hdu[1].header["camera"],
                        self.ccd == hdu[1].header["ccd"],
                    ]
                ):
                    raise ValueError("All files must have same sector, camera, ccd.")

            data = fitsio.read(fname)[:2048, 45 : 2048 + 45]
            k = (data > 0) & (data < 1500)
            # Blown out frames do not count.
            if (~k).sum() / (2048 ** 2) > 0.1:
                continue
            if diff is None:
                diff = data.copy()
            else:
                diff -= data
                diff_ar[fdx] = (np.abs(diff) > 200).sum() / (2048 ** 2)
                if (np.abs(diff) > 200).sum() / (2048 ** 2) > 0.002:
                    diff = data.copy()
                    continue
                diff = data.copy()
            if sat_mask is None:
                sat_mask = get_saturation_mask(data)
            grad = np.gradient(data)
            hard_mask |= np.hypot(*grad) > 60

            average[k] += data[k] - np.mean(data[k])
            weights[k] += 1
        average /= weights
        soft_mask = (np.hypot(*np.gradient(average)) > 30) | (weights == 0)
        soft_mask |= np.any(np.gradient(soft_mask.astype(float)), axis=0) != 0

        self.star_mask = ~(hard_mask | soft_mask | ~sat_mask)
        self.sat_mask = sat_mask

        # We don't need all these pixels, it's too many to store for every frame.
        # Instead we'll just save 5000 of them.
        s = np.random.choice((soft_mask & sat_mask).sum(), size=5000, replace=False)
        l = np.asarray(np.where(soft_mask & sat_mask))
        l = l[:, s]
        self.jitter_mask = np.zeros((2048, 2048), bool)
        self.jitter_mask[l[0], l[1]] = True

        #        with fits.open(self.fnames[self.reference_frame]) as hdu:
        fname = self.fnames[len(self.fnames) // 2]
        data = fitsio.read(fname)[:2048, 45 : 2048 + 45]
        grad = np.asarray(np.gradient(data))
        self.median_data = data[self.jitter_mask]
        self.median_gradient = grad[:, self.jitter_mask]

        return soft_mask, hard_mask, sat_mask, diff_ar

    def _build_matrices(self):
        """Allocate the matrices to fit the background.
        When we want to build the matrices to evaluate this background model,
        we will be able do so in a slightly more efficient way."""
        row, column = np.mgrid[:2048, :2048]
        c, r = column / 2048 - 0.5, row / 2048 - 0.5

        self._poly_X = np.asarray(
            [
                c.ravel() ** idx * r.ravel() ** jdx
                for idx in np.arange(self.npoly)
                for jdx in np.arange(self.npoly)
            ]
        ).T

        row, column = np.mgrid[: 2048 // self.nb, : 2048 // self.nb] * self.nb
        row, column = row + self.nb / 2, column + self.nb / 2
        c, r = column / 2048 - 0.5, row / 2048 - 0.5

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
                c.ravel() ** idx * r.ravel() ** jdx
                for idx in np.arange(self.npoly)
                for jdx in np.arange(self.npoly)
            ]
        ).T
        self.poly_sigma_w_inv = self._poly_X_down[self.weights.ravel() != 0].T.dot(
            self._poly_X_down[self.weights.ravel() != 0]
        )

        e = lil_matrix((2048, 2048 * 2048))
        for idx in range(2048):
            e[idx, np.arange(2048) * 2048 + idx] = 1
        self._strap_X = e.T.tocsr()
        self._spline_X = self._get_spline_matrix(np.arange(2048))
        self.X = hstack([self._spline_X, self._strap_X], format="csr")

        # We'll sacrifice some memory here for speed later.
        self.XT = self.X.T.tocsr()
        self.Xm = self.X[self.star_mask.ravel()].tocsr()
        self.XmT = self.X[self.star_mask.ravel()].T.tocsr()
        self.prior_mu = np.zeros(self._spline_X.shape[1] + self._strap_X.shape[1])
        self.prior_sigma = (
            np.ones(self._spline_X.shape[1] + self._strap_X.shape[1]) * 40
        )
        self.sigma_w_inv = self.XmT.dot(self.Xm) + np.diag(1 / self.prior_sigma ** 2)

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

        self.poly_w, self.spline_w, self.strap_w, self.t_start, self.jitter_pix = (
            np.zeros((len(self.fnames), self.npoly, self.npoly)),
            np.zeros((len(self.fnames), self.nknots, self.nknots)),
            np.zeros((len(self.fnames), 2048)),
            np.zeros(len(self.fnames)),
            np.zeros((len(self.fnames), self.jitter_mask.sum())),
        )

        for idx, fname in enumerate(tqdm(self.fnames, desc="Fitting FFI Frames")):
            (
                self.t_start[idx],
                self.poly_w[idx, :],
                self.spline_w[idx, :],
                self.strap_w[idx, :],
                self.jitter_pix[idx, :],
            ) = self._fit_frame(fname)
        # Smaller version of jitter for use later

        bad = sigma_clip(np.gradient(self.jitter_pix, axis=1).std(axis=1), sigma=5).mask
        _, med, std = sigma_clipped_stats(self.jitter_pix[~bad], axis=0)
        j = (np.copy(self.jitter_pix) - med) / std
        j[j > 10] = 0
        U, s, V = pca(j[~bad], 20, n_iter=100)
        X = np.zeros((self.jitter_pix.shape[0], U.shape[1]))
        X[~bad] = np.copy(U)
        self.jitter = X

    def _fit_frame(self, fname):
        """Helper function to fit a model to an individual frame."""
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

        # NOTE COLUMN NEEDS +45 EVENTUALLY
        data = fitsio.read(fname)[:2048, 45 : 2048 + 45]
        # data = hdu[1].data[:2048, 45 : 2048 + 45]
        t_start = hdu[0].header["TSTART"]

        avg = np.sum(
            [
                data[idx :: self.nb, jdx :: self.nb]
                * self.star_mask[idx :: self.nb, jdx :: self.nb]
                for idx in range(self.nb)
                for jdx in range(self.nb)
            ],
            axis=0,
        )

        avg /= self.weights
        B = self._poly_X_down[self.weights.ravel() != 0].T.dot(
            avg.ravel()[self.weights.ravel() != 0]
        )
        poly_w = np.linalg.solve(self.poly_sigma_w_inv, B)
        res = data - self._poly_X.dot(poly_w).reshape((2048, 2048))

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
            t_start,
            poly_w.reshape((self.npoly, self.npoly)),
            spline_w,
            strap_w,
            jitter_pix,
        )

    def _get_jitter(self, data):
        """Get the jitter correction somehow..."""
        # Use jitter mask.

        raise NotImplementedError

    def save(self):
        """
        Save a model fit to the tess-backrop data directory.

        Will create a fits file containing the following extensions
            - Primary
            - T_START: The time array for each background solution
            - KNOTS: Knot spacing in row and column
            - SPLINE_W: Solution to the spline model. Has shape (ntimes x nknots x nknots)
            - STRAP_W: Solution to the strap model. Has shape (ntimes x 2048)
            - POLY_W: Solution to the polynomial model. Has shape (ntimes x npoly x npoly)
        """
        if not hasattr(self, "star_mask"):
            raise ValueError(
                "It does not look like you have regenerated a tess_backdrop model, I do not think you want to save."
            )
        hdu0 = fits.PrimaryHDU()
        cols = [
            fits.Column(
                name="T_START", format="D", unit="BJD - 2457000", array=self.t_start
            ),
        ]
        hdu1 = fits.BinTableHDU.from_columns(cols)
        cols = [
            fits.Column(name="KNOTS", format="D", unit="PIX", array=self.knots_wbounds)
        ]
        hdu2 = fits.BinTableHDU.from_columns(cols)
        hdu3 = fits.ImageHDU(self.spline_w, name="spline_w")
        hdu4 = fits.ImageHDU(self.strap_w, name="strap_w")
        hdu5 = fits.ImageHDU(self.poly_w, name="poly_w")
        hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4, hdu5])
        hdul[0].header["ORIGIN"] = "tess-backdrop"
        hdul[0].header["AUTHOR"] = "christina.l.hedges@nasa.gov"
        hdul[0].header["VERSION"] = __version__

        for key in ["sector", "camera", "ccd", "nknots", "npoly", "degree"]:
            hdul[0].header[key] = getattr(self, key)

        fname = (
            f"tessbackdrop_sector{self.sector}_camera{self.camera}_ccd{self.ccd}.fits"
        )
        dir = f"{PACKAGEDIR}/data/sector{self.sector:03}/camera{self.camera:02}/ccd{self.ccd:02}/"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        hdul.writeto(dir + fname, overwrite=True)

        hdu0 = fits.PrimaryHDU()
        hdu1 = fits.ImageHDU(self.jitter, name="jitter_pix")
        hdul = fits.HDUList([hdu0, hdu1])
        hdul[0].header["ORIGIN"] = "tess-backdrop"
        hdul[0].header["AUTHOR"] = "christina.l.hedges@nasa.gov"
        hdul[0].header["VERSION"] = __version__
        for key in ["sector", "camera", "ccd", "nknots", "npoly", "degree"]:
            hdul[0].header[key] = getattr(self, key)
        fname = f"tessbackdrop_jitter_sector{self.sector}_camera{self.camera}_ccd{self.ccd}.fits"
        dir = f"{PACKAGEDIR}/data/sector{self.sector:03}/camera{self.camera:02}/ccd{self.ccd:02}/"
        hdul.writeto(dir + fname, overwrite=True)

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
            for key in ["sector", "camera", "ccd", "nknots", "npoly", "degree"]:
                setattr(self, key, hdu[0].header[key])
            self.t_start = hdu[1].data["T_START"]
            self.knots_wbounds = hdu[2].data["KNOTS"]
            self.spline_w = hdu[3].data
            self.strap_w = hdu[4].data
            self.poly_w = hdu[5].data

        fname = f"tessbackdrop_jitter_sector{sector}_camera{camera}_ccd{ccd}.fits"
        with fits.open(dir + fname, lazy_load_hdus=True) as hdu:
            self.jitter = hdu[1].data

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
        c, r = c / 2048 - 0.5, r / 2048 - 0.5

        self._poly_X = np.asarray(
            [
                c.ravel() ** idx * r.ravel() ** jdx
                for idx in np.arange(self.npoly)
                for jdx in np.arange(self.npoly)
            ]
        ).T
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


def get_saturation_mask(data, whisker_width=40):
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
    whisker_mask = np.zeros((2048, 2048), bool)
    for idx in np.arange(-2, 2):
        for jdx in np.arange(-whisker_width // 2, whisker_width // 2):

            a1 = np.max([np.zeros(len(centers)), centers[:, 1] - idx], axis=0)
            a1 = np.min([np.ones(len(centers)) * 2047, a1], axis=0)

            b1 = np.max([np.zeros(len(centers)), centers[:, 0] - jdx], axis=0)
            b1 = np.min([np.ones(len(centers)) * 2047, b1], axis=0)

            whisker_mask[a1.astype(int), b1.astype(int)] = True

    sat_mask = np.copy(sat_cols)
    sat_mask |= np.gradient(sat_mask.astype(float), axis=0) != 0
    for count in range(4):
        sat_mask |= np.gradient(sat_mask.astype(float), axis=1) != 0
    sat_mask |= whisker_mask

    X, Y = np.mgrid[:2048, :2048]

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
