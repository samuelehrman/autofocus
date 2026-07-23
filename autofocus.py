from scipy.ndimage import sobel
import numpy as np
from PIL import Image
from autoscript_sbd_microscope_client import SbdMicroscopeClient
from scipy.optimize import minimize_scalar

class Autofocus:
    def __init__(self, microscope: SbdMicroscopeClient):
        self.res = res
        self.hfw = hfw
        self.dwell = dwell
        self.simulating = simulating
        self.image_path = image_path
        self.wd = wd



    def set_simulating(self, simulating):
        self.simulating = simulating
        return self


    def get_simulated_image(self, wd, image_path="test.tif"):
        wd_ideal = 0.008

        image = Image.open(image_path)
        image = np.array(image)
        blur = np.abs(wd - wd_ideal) * 0.01
        image = np.gaussian_filter(image, blur)

    def get_microscope_image(self, wd):
        """
        Get an image from the microscope

        """
    
    class ImagingConditions:
        def __init__(self, res, hfw, dwell):
            self.res = res
            self.hfw = hfw
            self.dwell = dwell

        def set_imaging_conditions(self):
            """
            Set the imaging conditions on the microscope using self.res, self.hfw, and self.dwell.
            """

    def get_image(self, wd, simulating=self.simulating):
        if simulating:
            image = self.get_simulated_image(wd)
        else:
            image = self.get_microscope_image(wd)

    def get_metric(self, wd, simulating=self.simulating):
        image = self.get_image(wd, simulating=self.simulating)
        sharpness = self.sobel_variance(image)
        return sharpness



    ######################### Dumb Search Algorithms #########################

    def coarse_search(self, bounds, n_points):
        wds = np.linspace(bounds[0], bounds[1], n_points)
        iqs = np.array([self.get_metric(wd, simulating=self.simulating) for wd in wds])
        return wds, iqs

    def confidence_index(self, iqs):
        iqs = np.asarray(iqs, dtype=float)
        mean_iq = np.mean(iqs)
        if mean_iq == 0:
            return 0.0
        return float(np.max(iqs) / mean_iq)

    def refine_bounds_from_coarse(self, wds, iqs, search_bounds):
        """Window around the best coarse WD, clamped to the original search bounds."""
        wds = np.asarray(wds, dtype=float)
        iqs = np.asarray(iqs, dtype=float)
        spacing = wds[1] - wds[0] if len(wds) > 1 else 0.0
        best_wd = wds[np.argmax(iqs)]
        lo = max(search_bounds[0], best_wd - spacing)
        hi = min(search_bounds[1], best_wd + spacing)
        if lo >= hi:
            lo, hi = search_bounds[0], search_bounds[1]
        return (lo, hi)

    def convergent_search(self, bounds, max_iterations=15, tolerance=0.001):
        result = minimize_scalar(
            lambda wd: -self.get_metric(wd, simulating=self.simulating),
            bounds=bounds,
            method="bounded",
            options={"xatol": tolerance, "maxiter": max_iterations},
        )
        return result.x

    def quadratic_fit(self, bounds, n_points=5):
        wds, iqs = self.coarse_search(bounds, n_points)
        a, b, _c = np.polyfit(wds, iqs, 2)
        if a >= 0:
            return float(wds[np.argmax(iqs)])
        wd_peak = -b / (2 * a)
        return float(np.clip(wd_peak, bounds[0], bounds[1]))


    ######################### Algorithm Logic #########################

    def optimize_wd(self, bounds):
        alpha_LR_LHFW = 1.25
        alpha_LR_SHFW = 1.25
        factor_HR_LHFW = 1.25
        factor_HR_SHFW = 1.25

        LR_LHFW_coarse_imaging_conditions = self.ImagingConditions(res=self.res, hfw=self.hfw, dwell=self.dwell)
        LR_SHFW_coarse_imaging_conditions = self.ImagingConditions(res=self.res, hfw=self.hfw / 50, dwell=self.dwell)
        HR_LHFW_coarse_imaging_conditions = self.ImagingConditions(res=self.res, hfw=self.hfw, dwell=self.dwell * 2)
        HR_SHFW_coarse_imaging_conditions = self.ImagingConditions(res=self.res, hfw=self.hfw / 50, dwell=self.dwell * 2)

        LR_LHFW_convergent_imaging_conditions = self.ImagingConditions(res=self.res, hfw=self.hfw, dwell=self.dwell * 2)
        LR_SHFW_convergent_imaging_conditions = self.ImagingConditions(res=self.res, hfw=self.hfw / 50, dwell=self.dwell * 2)
        HR_LHFW_convergent_imaging_conditions = self.ImagingConditions(res=self.res, hfw=self.hfw, dwell=self.dwell * 4)
        HR_SHFW_convergent_imaging_conditions = self.ImagingConditions(res=self.res, hfw=self.hfw / 50, dwell=self.dwell * 4)

        # Stage 1: quick large-HFW coarse search
        LR_LHFW_coarse_imaging_conditions.set_imaging_conditions()
        wds_LR_LHFW, iqs_LR_LHFW = self.coarse_search(bounds, n_points=self.n_points_LR_LHFW)
        CI_LR_LHFW = self.confidence_index(iqs_LR_LHFW)

        if CI_LR_LHFW > alpha_LR_LHFW:
            convergent_bounds = self.refine_bounds_from_coarse(wds_LR_LHFW, iqs_LR_LHFW, bounds)
            LR_LHFW_convergent_imaging_conditions.set_imaging_conditions()

            ## future work -> check if results converge and if not go to next stage

            return self.convergent_search(
                convergent_bounds,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
            )



        # Stage 2: quick small-HFW coarse search
        LR_SHFW_coarse_imaging_conditions.set_imaging_conditions()
        wds_LR_SHFW, iqs_LR_SHFW = self.coarse_search(bounds, n_points=self.n_points_LR_SHFW)
        CI_LR_SHFW = self.confidence_index(iqs_LR_SHFW)

        if CI_LR_SHFW > alpha_LR_SHFW:
            convergent_bounds = self.refine_bounds_from_coarse(wds_LR_SHFW, iqs_LR_SHFW, bounds)
            LR_SHFW_convergent_imaging_conditions.set_imaging_conditions()
            return self.convergent_search(
                convergent_bounds,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
            )

        # Stage 3: higher dwell / S/N at both HFWs; refine with the better signal
        HR_LHFW_coarse_imaging_conditions.set_imaging_conditions()
        wds_HR_LHFW, iqs_HR_LHFW = self.coarse_search(bounds, n_points=self.n_points_LS_LHFW)
        CI_HR_LHFW = self.confidence_index(iqs_HR_LHFW)

        HR_SHFW_coarse_imaging_conditions.set_imaging_conditions()
        wds_HR_SHFW, iqs_HR_SHFW = self.coarse_search(bounds, n_points=self.n_points_LS_SHFW)
        CI_HR_SHFW = self.confidence_index(iqs_HR_SHFW)

        relative_quality_HR_LHFW = CI_HR_LHFW * factor_HR_LHFW
        relative_quality_HR_SHFW = CI_HR_SHFW * factor_HR_SHFW

        if relative_quality_HR_LHFW > relative_quality_HR_SHFW:
            convergent_bounds = self.refine_bounds_from_coarse(wds_HR_LHFW, iqs_HR_LHFW, bounds)
            HR_LHFW_convergent_imaging_conditions.set_imaging_conditions()
        else:
            convergent_bounds = self.refine_bounds_from_coarse(wds_HR_SHFW, iqs_HR_SHFW, bounds)
            HR_SHFW_convergent_imaging_conditions.set_imaging_conditions()

        return self.convergent_search(
            convergent_bounds,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )


    ######################### Image Quality Metrics #########################

    def sobel_variance(self, image):
        image = image.astype(np.float64)
        sobel_x = sobel(image, axis=0)
        sobel_y = sobel(image, axis=1)
        variance = np.var(np.hypot(sobel_x, sobel_y))
        return variance

    def std(self, image):
        image = image.astype(np.float64)
        return np.std(image)

    def FFT_power_above_thresh(self, image, freq_thresh=0.1):
        """High-frequency energy fraction. freq_thresh in cycles/pixel, in (0, 0.5)."""
        image = image.astype(np.float64)
        image = image - image.mean()  # reduces DC dominance before normalize
        power = np.abs((np.fft.fft2(image)))**2
        power /= power.sum()  # normalize so all coefficients sum to 1
        fy = np.fft.fftfreq(image.shape[0])  # cycles/pixel
        fx = np.fft.fftfreq(image.shape[1])
        radius = np.sqrt(fy[:, None]**2 + fx[None, :]**2)
        return power[radius > freq_thresh].sum()


