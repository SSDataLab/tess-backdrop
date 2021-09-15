import logging
from functools import lru_cache

import fitsio
import matplotlib.pyplot as plt
import numpy as np
from lightkurve.correctors.designmatrix import _spline_basis_vector
from matplotlib import animation
from scipy.sparse import csr_matrix, hstack, lil_matrix, vstack


log = logging.getLogger(__name__)

__all__ = [
    #    "get_knots",
    "find_saturation_column_centers",
    #    "find_bad_frames",
    "get_saturation_mask",
    "std_iter",
    "animate",
    #    "get_spline_matrix",
]


@lru_cache(maxsize=16)
def _flux(fname, cutout_size=2048, limit=100):
    """Let's us cache some frames in memory"""
    f = fitsio.read(fname)[:cutout_size, 45 : 45 + cutout_size]
    f[f < limit] = np.nan
    return f


def _bin_down(flux, nb, cutout_size=2048, func=np.nanmin, limit=100):
    """Bins the flux down to a resolution of 2048/`nb` and takes the `func`"""
    if nb == 1:
        return flux
    ar = np.zeros((cutout_size // nb, cutout_size // nb, nb, nb))
    for idx in range(nb):
        for jdx in range(nb):
            ar[:, :, idx, jdx] = flux[idx::nb, jdx::nb]
    ar[ar <= 0] = np.nan
    return func(ar, axis=(2, 3))


def _X(x, knots, degree):
    """Makes a matrix of spline components from a vectors

    Parameters
    ----------
    x : np.ndarray
        Vector to create a matrix from
    knots: np.ndarray
        Array of knots WITH BOUNDS
    degree : int
        Degree of spline

    Returns
    -------
    X : np.ndarray
        Matrix of spline components
    """
    matrices = [
        csr_matrix(_spline_basis_vector(x, degree, idx, knots))
        for idx in np.arange(-1, len(knots) - degree - 1)
    ]
    X = vstack(matrices, format="csr").T
    return X


def plot(self, frame=0, vmin=None, vmax=None):
    """Plots a given frame of the data, model and residuals"""
    with plt.style.context("seaborn-white"):
        f = self.flux(frame)
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


def correct_tpf(self, tpf, exptime=None):
    """Returns a TPF with the background corrected

    Parameters
    ----------
    self : tbd.SimpleBackDrop or tbd.FullBackDrop
        A backdrop object to use to correct the TPF
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


def animate(data, scale="linear", output="out.mp4", **kwargs):
    """Create an animation of all the frames in `data`.
    Parameters
    ----------
    data : np.ndarray
        3D np.ndarray
    output : str
        File to output mp4 to
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    idx = 0
    if scale is "log":
        dat = np.log10(np.copy(data))
    else:
        dat = data
    cmap = kwargs.pop("cmap", "Greys_r")
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad("black")
    if "vmax" not in kwargs:
        kwargs["vmin"] = np.nanpercentile(dat, 70)
        kwargs["vmax"] = np.nanpercentile(dat, 99.9)
    im = ax.imshow(dat[idx], origin="lower", cmap=cmap, **kwargs)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.axis("off")

    def animate(idx):
        im.set_data(dat[idx])
        return (im,)

    anim = animation.FuncAnimation(
        fig, animate, frames=len(dat), interval=30, blit=True
    )
    anim.save(output, dpi=150)


#
# def get_spline_matrix(x1, knots1, x2=None, knots2=None, degree=2):
#     """Helper function to make a 2D spline matrix in a fairly memory efficient way."""
#     if x1 is None:
#         x2 = x1
#         knots2 = knots1
#
#     def _X(x, knots, degree):
#         matrices = [
#             csr_matrix(_spline_basis_vector(x, degree, idx, knots))
#             for idx in np.arange(-1, len(knots) - degree - 1)
#         ]
#         X = vstack(matrices, format="csr").T
#         return X
#
#     X1 = _X(x1, knots1, degree)
#     X2 = _X(x2, knots2, degree)
#     X1f = hstack([X1 for idx in range(X2.shape[1])]).tocsr()
#     X2f = vstack([X2 for idx in range(X1.shape[1])])
#     X2f = X2f.reshape(X1f.shape).tocsr()
#     Xf = X1f.multiply(X2f)
#     return Xf


# def get_knots(x, nknots, degree):
#     """Find the b-spline knot spacing for an input array x, number of knots and degree
#
#     Parameters
#     ----------
#     x : np.ndarray
#         In put vector to create b-spline for
#     nknots : int
#         Number of knots to use in the b-spline
#     degree : int
#         Degree of the b-spline
#
#     Returns
#     -------
#     knots_wbounds : np.ndarray
#         The knot locations for the input x.
#     """
#     x = np.sort(x)
#     knots = np.asarray(
#         [s[-1] for s in np.array_split(np.argsort(x), nknots - degree)[:-1]]
#     )
#     knots = [np.mean([x[k], x[k + 1]]) for k in knots]
#     knots = np.append(np.append(x.min(), knots), x.max())
#     knots = np.unique(knots)
#     knots_wbounds = np.append(
#         np.append([x.min()] * (degree - 1), knots), [x.max()] * (degree)
#     )
#     return knots_wbounds


def find_saturation_column_centers(mask):
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
    if not sat_cols.any():
        return ~sat_cols

    centers, radii = find_saturation_column_centers(sat_cols)
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


def std_iter(x, mask, sigma=3, n_iters=3):
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


# def find_bad_frames(fnames, cutout_size=2048, corner_check=False):
#     """Identifies frames that probably have a lot of scattered lightkurve
#     If quality flags are available, will use TESS quality flags.
#
#     If unavailable, or if `corner_check=True`, loads the 30x30 pixel corner
#     region of every frame, and uses them to find frames that have a lot of
#     scattered light.
#
#
#     """
#
#     quality = np.zeros(len(fnames), int)
#     warned = False
#
#     log.info("Extracting quality")
#     for idx, fname in enumerate(fnames):
#         try:
#             quality[idx] = fitsio.read_header(fname, 1)["DQUALITY"]
#         except KeyError:
#             if warned is False:
#                 log.warning("Quality flags are missing.")
#                 warned = True
#             continue
#     bad = (quality & (2048 | 175)) != 0
#
#     if warned | corner_check:
#         log.info("Using corner check")
#         corner = np.zeros((4, len(fnames)))
#         for tdx, fname in enumerate(fnames):
#             corner[0, tdx] = fitsio.read(fname)[:30, 45 : 45 + 30].mean()
#             corner[1, tdx] = fitsio.read(fname)[-30:, 45 : 45 + 30].mean()
#             corner[2, tdx] = fitsio.read(fname)[
#                 :30, 45 + cutout_size - 30 : 45 + cutout_size
#             ].mean()
#             corner[3, tdx] = fitsio.read(fname)[
#                 -30:, 45 + cutout_size - 30 - 1 : 45 + cutout_size
#             ].mean()
#
#         c = corner.T - np.median(corner, axis=1)
#         c /= np.std(c, axis=0)
#         bad = (np.abs(c) > 2).any(axis=1)
#         #    bad |= corner.std(axis=0) > 200
#     return bad, quality


# def get_spline_matrix(xc, xr=None, degree=2, nknots=20):
#     """Helper function to make a 2D spline matrix in a fairly memory efficient way."""
#
#     def _X(x, degree, knots):
#         matrices = [
#             csr_matrix(_spline_basis_vector(x, degree, idx, knots))
#             for idx in np.arange(-1, len(knots) - degree - 1)
#         ]
#         X = vstack(matrices, format="csr").T
#         return X
#
#     xc_knots = get_knots(xc, nknots=nknots, degree=degree)
#     if xr is None:
#         xr = xc
#         xr_knots = xc_knots
#     else:
#         xr_knots = get_knots(xr, nknots=nknots, degree=degree)
#     Xc = _X(xc, degree, xc_knots)
#     Xcf = vstack([Xc for idx in range(len(xr))]).tocsr()
#     Xr = _X(xr, degree, xr_knots)
#     Xrf = (
#         hstack([Xr for idx in range(len(xc))])
#         .reshape((Xcf.shape[0], Xc.shape[1]))
#         .tocsr()
#     )
#     Xf = hstack([Xrf.multiply(X.T) for X in Xcf.T]).tocsr()
#     return Xf
