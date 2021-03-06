"""Corrector function for TESSCut TPFs"""
import lightkurve as lk
import numpy as np

from .backdrop import BackDrop


class TESSCutCorrector(lk.RegressionCorrector):
    """Remove TESS jitter and sky background noise using linear regression.

    Will automatically generate a design matrix based on `tess_backdrop` stored files.

    Parameters
    ----------
    tpf : `lightkurve.TargetPixelFile`
        The target pixel file for a target
    aperture_mask : np.ndarray of booleans
        Aperture mask to apply to TPF. If none, one will be
        selected per `lightkurve` defaults.
    """

    def __init__(self, tpf, aperture_mask=None):
        """
        Parameters
        ----------
        tpf : `lightkurve.TargetPixelFile`
            The target pixel file for a target
        aperture_mask : np.ndarray of booleans
            Aperture mask to apply to TPF. If none, one will be
            selected per `lightkurve` defaults.
        """
        if aperture_mask is None:
            aperture_mask = tpf.create_threshold_mask(3)
        self.aperture_mask = aperture_mask
        lc = tpf.to_lightcurve(aperture_mask=aperture_mask)
        # Remove cadences that have NaN flux (cf. #874). We don't simply call
        # `lc.remove_nans()` here because we need to mask both lc & tpf.
        nan_mask = np.isnan(lc.flux)
        lc = lc[~nan_mask]
        self.b = BackDrop()
        self.b.load(tpf.sector, tpf.camera, tpf.ccd)
        self.tpf = self.b.correct_tpf(tpf)[~nan_mask]
        self.lc = self.tpf.to_lightcurve(aperture_mask=aperture_mask)
        super().__init__(lc=self.lc)

    def __repr__(self):
        if self.lc.label == "":
            return "TESSCutCorrector (ID: {})".format(self.lc.targetid)
        return "TESSCutCorrector (ID: {})".format(self.lc.label)

    def correct(
        self,
        cadence_mask=None,
        sigma=5,
        niters=3,
        propagate_errors=False,
        spline_timescale=0.5,
        spline_degree=3,
        npca_components=10,
    ):
        """Returns a systematics-corrected light curve from a TESSCut TPF.

        Parameters
        ----------
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.
        sigma : int (default 5)
            Standard deviation at which to remove outliers from fitting
        niters : int (default 5)
            Number of iterations to fit and remove outliers
        propagate_errors : bool (default False)
            Whether to propagate the uncertainties from the regression. Default is False.
            Setting to True will increase run time, but will sample from multivariate normal
            distribution of weights.
        spline_timescale : float, None
            Time between knots in spline component. If None, will not use spline.
        spline_degree : int
            Polynomial degree of spline.
        npca_components : int, default 10
            Number of terms added to the design matrix for jitter correction.


        Returns
        -------
        clc : `lightkurve.LightCurve`
            Systematics-corrected `lightkurve.LightCurve`.
        """

        bad = ~lk.utils.TessQualityFlags.create_quality_mask(
            self.lc.quality,
            self.tpf.quality & lk.utils.TessQualityFlags.DEFAULT_BITMASK,
        )
        med_flux = np.median(self.lc.flux[cadence_mask & ~bad].value)
        if spline_timescale is not None:
            # Spline DM
            knots = np.linspace(
                self.lc.time[0].value,
                self.lc.time[-1].value,
                int(
                    (self.lc.time[-1].value - self.lc.time[0].value) / spline_timescale
                ),
            )[1:-1]
            dm_spline = lk.designmatrix.create_sparse_spline_matrix(
                self.lc.time.value, knots=knots, degree=spline_degree
            )
            dm_spline.prior_mu = np.zeros(dm_spline.shape[1]) * med_flux
            dm_spline.prior_sigma = np.ones(dm_spline.shape[1]) * med_flux * 3
        dm_ones = lk.DesignMatrix(
            np.ones(len(self.lc.flux))[:, None],
            name="ones",
            prior_mu=[med_flux],
            prior_sigma=[0.1 * med_flux],
        )

        # Scattered Light DM
        bkg = self.b.build_correction(
            np.arange(self.tpf.shape[2]) + self.tpf.column,
            np.arange(self.tpf.shape[1]) + self.tpf.row,
        )
        bkg = bkg[:, self.aperture_mask].sum(axis=1)
        bkg -= np.median(bkg)
        dm_bkg = lk.DesignMatrix(
            np.vstack([bkg ** idx for idx in np.arange(1, 4)]).T,
            name="sky",
            prior_mu=[0, 0, 0],
            prior_sigma=[
                self.lc.flux.value[~bad].std(),
                self.lc.flux.value[~bad].std(),
                self.lc.flux.value[~bad].std(),
            ],
        )

        # Jitter DM
        dm_jitter = lk.DesignMatrix(
            self.b.jitter_comps[:, : npca_components * 3],
            name="jitter",
            prior_mu=np.zeros(npca_components * 3),
            prior_sigma=np.ones(npca_components * 3) * self.lc.flux.value.std() * 3,
        )

        breaks = (
            np.where(np.diff(self.b.t_start) > np.median(np.diff(self.b.t_start) * 10))[
                0
            ]
            + 1
        )
        if spline_timescale is not None:
            dm = lk.SparseDesignMatrixCollection(
                [
                    dm_bkg.to_sparse(),
                    dm_jitter.to_sparse(),
                    dm_spline,
                    dm_ones.to_sparse(),
                ]
            ).split(list(breaks))
        else:
            dm = lk.SparseDesignMatrixCollection(
                [
                    dm_bkg.to_sparse(),
                    dm_jitter.to_sparse(),
                    dm_ones.to_sparse(),
                ]
            ).split(list(breaks))

        if cadence_mask is None:
            cadence_mask = np.ones(len(self.lc.time), bool)
        super().correct(
            dm,
            cadence_mask=cadence_mask & ~bad,
            sigma=sigma,
            niters=niters,
            propagate_errors=propagate_errors,
        )
        # clc += self.diagnostic_lightcurves["spline"]
        # clc -= np.median(clc.flux)
        # clc += np.percentile(self.lc.flux, 10)
        return (self.lc.copy() - self.diagnostic_lightcurves["sky"]) / (
            self.diagnostic_lightcurves["ones"] + self.diagnostic_lightcurves["jitter"]
        )
