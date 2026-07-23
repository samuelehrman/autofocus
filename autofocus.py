import time

from scipy.ndimage import sobel, gaussian_filter
import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar

try:
    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client import enumerations as enums
    from autoscript_sdb_microscope_client import structures as structs

    RESOLUTION_PRESETS = {
        6144: enums.ScanningResolution.PRESET_6144X4096,
        3072: enums.ScanningResolution.PRESET_3072X2048,
        1536: enums.ScanningResolution.PRESET_1536X1024,
        768: enums.ScanningResolution.PRESET_768X512,
    }
except ImportError:
    SdbMicroscopeClient = object
    enums = None
    structs = None
    RESOLUTION_PRESETS = {}


class Autofocus:
    def __init__(
        self,
        microscope: SdbMicroscopeClient = None,
        res=1536,
        hfw=1e-3,
        dwell=1e-6,
        simulating=False,
        image_path="simulated_image/test.tif",
        n_points_LR_LHFW=8,
        n_points_LR_SHFW=8,
        n_points_LS_LHFW=16,
        n_points_LS_SHFW=16,
        max_iterations=15,
        tolerance=1e-5,
        bit_depth=8,
        beam="electron",
    ):
        self.microscope = microscope
        self.res = res
        self.hfw = hfw
        self.dwell = dwell
        self.simulating = simulating
        self.image_path = image_path
        self.n_points_LR_LHFW = n_points_LR_LHFW
        self.n_points_LR_SHFW = n_points_LR_SHFW
        self.n_points_LS_LHFW = n_points_LS_LHFW
        self.n_points_LS_SHFW = n_points_LS_SHFW
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.bit_depth = bit_depth
        self.beam = beam

    def set_simulating(self, simulating):
        self.simulating = simulating
        return self

    def _beam(self, beam=None):
        beam = (beam or self.beam).lower()
        if beam == "electron":
            return self.microscope.beams.electron_beam
        if beam == "ion":
            return self.microscope.beams.ion_beam
        raise ValueError(f"Invalid beam: {beam}. Use 'electron' or 'ion'.")

    def _resolve_resolution(self, resolution):
        if resolution is None:
            return None
        if isinstance(resolution, float):
            resolution = int(resolution)
        elif isinstance(resolution, str):
            resolution = int(resolution.strip())
        if resolution in RESOLUTION_PRESETS:
            return RESOLUTION_PRESETS[resolution]
        raise ValueError(
            f"Invalid resolution: {resolution}. Use one of {sorted(RESOLUTION_PRESETS)}."
        )

    ######################### Microscope Imaging Controls #########################

    def get_resolution(self, beam=None):
        return self._beam(beam).scanning.resolution.value

    def set_resolution(self, resolution, beam=None):
        resolved = self._resolve_resolution(resolution)
        self._beam(beam).scanning.resolution.value = resolved

    def get_dwell(self, beam=None):
        return self._beam(beam).scanning.dwell_time.value

    def set_dwell(self, dwell_time, beam=None):
        self._beam(beam).scanning.dwell_time.value = dwell_time

    def get_hfw(self, beam=None):
        return self._beam(beam).horizontal_field_width.value

    def set_hfw(self, hfw, beam=None):
        self._beam(beam).horizontal_field_width.value = hfw

    def get_wd(self, beam=None):
        return self._beam(beam).working_distance.value

    def set_wd(self, wd, beam=None):
        self._beam(beam).working_distance.value = wd
        time.sleep(0.1)

    def prepare_imaging(
        self,
        resolution=None,
        dwell_time=None,
        hfw=None,
        tilt_correction=False,
        dynamic_focus=False,
        reduced_area=None,
        beam=None,
    ):
        """Apply imaging settings on the microscope (pattern from notes.txt)."""
        if self.simulating or self.microscope is None:
            return True

        beam_obj = self._beam(beam)

        if resolution is not None:
            self.set_resolution(resolution, beam=beam)

        if dwell_time is not None:
            self.set_dwell(dwell_time, beam=beam)

        if hfw is not None:
            self.set_hfw(hfw, beam=beam)

        # Angular correction exists on electron beam in the notes example
        if hasattr(beam_obj, "angular_correction"):
            if tilt_correction:
                beam_obj.angular_correction.tilt_correction.turn_on()
            else:
                beam_obj.angular_correction.tilt_correction.turn_off()
            if dynamic_focus:
                beam_obj.angular_correction.dynamic_focus.turn_on()
            else:
                beam_obj.angular_correction.dynamic_focus.turn_off()

        if reduced_area is not None:
            beam_obj.scanning.mode.set_reduced_area(*reduced_area)
        else:
            beam_obj.scanning.mode.set_full_frame()

        return True

    def grab_frame(self, bit_depth=None):
        bit_depth = self.bit_depth if bit_depth is None else bit_depth
        frame = self.microscope.imaging.grab_frame(
            structs.GrabFrameSettings(bit_depth=bit_depth)
        )
        return np.copy(frame.data)

    def get_simulated_image(self, wd, image_path=None):
        image_path = image_path or self.image_path
        wd_ideal = 0.008

        image = Image.open(image_path)
        image = np.array(image, dtype=np.float64)
        # Map WD error (m) to blur sigma (pixels): 1 mm defocus -> sigma ~5
        blur = np.abs(wd - wd_ideal) / 0.001 * 5.0
        if blur > 0:
            image = gaussian_filter(image, blur)
        return image

    def get_microscope_image(self, wd):
        """Set working distance and grab a frame from the microscope."""
        self.set_wd(wd)
        return self.grab_frame()

    class ImagingConditions:
        def __init__(self, autofocus, res, hfw, dwell):
            self.autofocus = autofocus
            self.res = res
            self.hfw = hfw
            self.dwell = dwell

        def set_imaging_conditions(self):
            """Apply res / hfw / dwell on the microscope."""
            self.autofocus.prepare_imaging(
                resolution=self.res,
                dwell_time=self.dwell,
                hfw=self.hfw,
            )

    def get_image(self, wd, simulating=None):
        if simulating is None:
            simulating = self.simulating
        if simulating:
            return self.get_simulated_image(wd)
        return self.get_microscope_image(wd)

    def get_metric(self, wd, simulating=None):
        if simulating is None:
            simulating = self.simulating
        image = self.get_image(wd, simulating=simulating)
        return self.sobel_variance(image)

    ######################### Dumb Search Algorithms #########################

    def coarse_search(self, bounds, n_points):
        wds = np.linspace(bounds[0], bounds[1], n_points)
        iqs = np.array([self.get_metric(wd) for wd in wds])
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
            lambda wd: -self.get_metric(wd),
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

        LR_LHFW_coarse_imaging_conditions = self.ImagingConditions(
            self, res=self.res, hfw=self.hfw, dwell=self.dwell
        )
        LR_SHFW_coarse_imaging_conditions = self.ImagingConditions(
            self, res=self.res, hfw=self.hfw / 50, dwell=self.dwell
        )
        HR_LHFW_coarse_imaging_conditions = self.ImagingConditions(
            self, res=self.res, hfw=self.hfw, dwell=self.dwell * 2
        )
        HR_SHFW_coarse_imaging_conditions = self.ImagingConditions(
            self, res=self.res, hfw=self.hfw / 50, dwell=self.dwell * 2
        )

        LR_LHFW_convergent_imaging_conditions = self.ImagingConditions(
            self, res=self.res, hfw=self.hfw, dwell=self.dwell * 2
        )
        LR_SHFW_convergent_imaging_conditions = self.ImagingConditions(
            self, res=self.res, hfw=self.hfw / 50, dwell=self.dwell * 2
        )
        HR_LHFW_convergent_imaging_conditions = self.ImagingConditions(
            self, res=self.res, hfw=self.hfw, dwell=self.dwell * 4
        )
        HR_SHFW_convergent_imaging_conditions = self.ImagingConditions(
            self, res=self.res, hfw=self.hfw / 50, dwell=self.dwell * 4
        )

        # Stage 1: quick large-HFW coarse search
        LR_LHFW_coarse_imaging_conditions.set_imaging_conditions()
        wds_LR_LHFW, iqs_LR_LHFW = self.coarse_search(
            bounds, n_points=self.n_points_LR_LHFW
        )
        CI_LR_LHFW = self.confidence_index(iqs_LR_LHFW)

        if CI_LR_LHFW > alpha_LR_LHFW:
            convergent_bounds = self.refine_bounds_from_coarse(
                wds_LR_LHFW, iqs_LR_LHFW, bounds
            )
            LR_LHFW_convergent_imaging_conditions.set_imaging_conditions()

            ## future work -> check if results converge and if not go to next stage

            return self.convergent_search(
                convergent_bounds,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
            )

        # Stage 2: quick small-HFW coarse search
        LR_SHFW_coarse_imaging_conditions.set_imaging_conditions()
        wds_LR_SHFW, iqs_LR_SHFW = self.coarse_search(
            bounds, n_points=self.n_points_LR_SHFW
        )
        CI_LR_SHFW = self.confidence_index(iqs_LR_SHFW)

        if CI_LR_SHFW > alpha_LR_SHFW:
            convergent_bounds = self.refine_bounds_from_coarse(
                wds_LR_SHFW, iqs_LR_SHFW, bounds
            )
            LR_SHFW_convergent_imaging_conditions.set_imaging_conditions()
            return self.convergent_search(
                convergent_bounds,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
            )

        # Stage 3: higher dwell / S/N at both HFWs; refine with the better signal
        HR_LHFW_coarse_imaging_conditions.set_imaging_conditions()
        wds_HR_LHFW, iqs_HR_LHFW = self.coarse_search(
            bounds, n_points=self.n_points_LS_LHFW
        )
        CI_HR_LHFW = self.confidence_index(iqs_HR_LHFW)

        HR_SHFW_coarse_imaging_conditions.set_imaging_conditions()
        wds_HR_SHFW, iqs_HR_SHFW = self.coarse_search(
            bounds, n_points=self.n_points_LS_SHFW
        )
        CI_HR_SHFW = self.confidence_index(iqs_HR_SHFW)

        relative_quality_HR_LHFW = CI_HR_LHFW * factor_HR_LHFW
        relative_quality_HR_SHFW = CI_HR_SHFW * factor_HR_SHFW

        if relative_quality_HR_LHFW > relative_quality_HR_SHFW:
            convergent_bounds = self.refine_bounds_from_coarse(
                wds_HR_LHFW, iqs_HR_LHFW, bounds
            )
            HR_LHFW_convergent_imaging_conditions.set_imaging_conditions()
        else:
            convergent_bounds = self.refine_bounds_from_coarse(
                wds_HR_SHFW, iqs_HR_SHFW, bounds
            )
            HR_SHFW_convergent_imaging_conditions.set_imaging_conditions()

        return self.convergent_search(
            convergent_bounds,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )

    def find_optimal_wd(self, bounds):
        """Run the staged optimizer, set the result on the microscope, and return it."""
        wd = self.optimize_wd(bounds)
        if not self.simulating:
            self.set_wd(wd)
        return wd

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
        power = np.abs(np.fft.fft2(image)) ** 2
        power /= power.sum()  # normalize so all coefficients sum to 1
        fy = np.fft.fftfreq(image.shape[0])  # cycles/pixel
        fx = np.fft.fftfreq(image.shape[1])
        radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        return power[radius > freq_thresh].sum()
