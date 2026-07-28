import time

from scipy.ndimage import sobel, gaussian_filter
import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from scipy.stats import stats
from scipy.signal import find_peaks

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
        testing=False,
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
        self.testing = testing
        self._testing_wds: list = []
        self._testing_iqs: list = []

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
        self._beam(beam).working_distance.set_value_no_degauss(wd)
        time.sleep(0.01)

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
        t0 = time.perf_counter()
        self.set_wd(wd)
        t1 = time.perf_counter()
        frame = self.grab_frame()
        t2 = time.perf_counter()
        print(f"        set_wd={t1-t0:.3f}s  grab_frame={t2-t1:.3f}s")
        return frame

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

    def compute_metric(self, image):
        """Compute the sharpness metric from an already-acquired image."""
        return self.FFT_power_above_thresh(image)

    def get_metric(self, wd, simulating=None):
        if simulating is None:
            simulating = self.simulating
        image = self.get_image(wd, simulating=simulating)
        metric = self.compute_metric(image)
        if self.testing:
            self._testing_wds.append(wd)
            self._testing_iqs.append(metric)
        return metric

        ######################### Dumb Search Algorithms #########################

    def coarse_search(self, bounds, n_points):
        wds = np.linspace(bounds[0], bounds[1], n_points)

        # Phase 1: acquire all images without processing
        images = []
        t_acq_start = time.perf_counter()
        for i, wd in enumerate(wds):
            t0 = time.perf_counter()
            print(f"    [{i+1}/{n_points}] Acquiring image at WD = {wd*1e3:.4f} mm ...", flush=True)
            images.append(self.get_image(wd))
            print(f"      -> acquired in {time.perf_counter()-t0:.3f} s")
        t_acq_total = time.perf_counter() - t_acq_start
        print(f"  Acquisition complete: {n_points} images in {t_acq_total:.3f} s "
              f"({t_acq_total/n_points:.3f} s/image)")

        # Phase 2: process the image stack
        print(f"  Processing {n_points}-image stack ...")
        t_proc_start = time.perf_counter()
        iqs = []
        for i, (wd, image) in enumerate(zip(wds, images)):
            t0 = time.perf_counter()
            metric = self.compute_metric(image)
            if self.testing:
                self._testing_wds.append(wd)
                self._testing_iqs.append(metric)
            iqs.append(metric)
            print(f"    [{i+1}/{n_points}] WD = {wd*1e3:.4f} mm -> sharpness = {metric:.4f} "
                  f"({time.perf_counter()-t0:.3f} s)")
        t_proc_total = time.perf_counter() - t_proc_start
        print(f"  Processing complete: {t_proc_total:.3f} s total "
              f"({t_proc_total/n_points:.3f} s/image)")

        return wds, np.array(iqs)

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
        print(f"  Convergent search in [{bounds[0]*1e3:.4f}, {bounds[1]*1e3:.4f}] mm "
              f"(tol={tolerance*1e3:.4f} mm, max_iter={max_iterations}) ...")
        _eval_count = [0]
        _history = [] # (wd, metric)

        def _neg_metric(wd):
            _eval_count[0] += 1
            t0 = time.perf_counter()
            print(f"    [iter {_eval_count[0]}] WD = {wd*1e3:.4f} mm ...", end=" ", flush=True)
            m = self.get_metric(wd)
            print(f"sharpness = {m:.4f}  ({time.perf_counter()-t0:.3f} s)")
            _history.append((wd, m))
            return -m

        t_conv_start = time.perf_counter()
        result = minimize_scalar(
            _neg_metric,
            bounds=bounds,
            method="bounded",
            options={"xatol": tolerance, "maxiter": max_iterations},
        )

        # Also get the best five metrics
        best5 = sorted(_history, key=lambda p: p[1], reverse=True)[:5]
        best5_wds, best5_iqs = zip(*best5) if best5 else ((), ())

        print(f"  Convergent search done in {time.perf_counter()-t_conv_start:.3f} s.  "
              f"Best WD = {result.x*1e3:.4f} mm")
        return result.x, best5_wds, best5_iqs

    def laplacian_fit(self, wds, iqs):
        # Fit the image metric to a Laplacian
        loc, scale = stats.laplace.fit(iqs)
        print(f"Fitted Location (mu): {loc:.4f}")
        print(f"Fitted Scale (b): {scale:.4f}")

        # Simulate data in the metric and WD space
        metric_range = np.linspace(min(iqs), max(iqs), 200)
        wd_range = np.linspace(min(wds), max(iqs), 200)

        # Fit a probability distribution function
        pdf_values = stats.laplace.pdf(metric_range, loc, scale)

        # Find where the peak occurs in metric_range (returns a location)
        peak_idx, _ = find_peaks(pdf_values)

        # Best values
        print(f"  Best metric value: {metric_range[peak_idx]:.4f}")
        print(f"  Best WD value: {wds[peak_idx]*1e3:.4f} mm")
        return float(wds[peak_idx]*1e3)

    ######################### Algorithm Logic #########################

    def optimize_wd(self, bounds):
        alpha_LR_LHFW = 1.1
        alpha_LR_SHFW = 1.1
        factor_HR_LHFW = 1.1
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

        # Stage 1: Fine search since we should already be close to focused
        # higher dwell / S/N at both HFWs; refine with the better signal
        print(f"\n[Stage 1] High-SNR coarse scan at both HFWs ...")
        print(f"  [Stage 1a] Large HFW = {self.hfw * 1e3:.3f} mm, {self.n_points_LS_LHFW} points ...")
        HR_LHFW_coarse_imaging_conditions.set_imaging_conditions()
        wds_HR_LHFW, iqs_HR_LHFW = self.coarse_search(
            bounds, n_points=self.n_points_LS_LHFW
        )
        CI_HR_LHFW = self.confidence_index(iqs_HR_LHFW)
        print(f"  Confidence index (large HFW) = {CI_HR_LHFW:.3f}")

        print(f"  [Stage 1b] Small HFW = {self.hfw / 50 * 1e3:.3f} mm, {self.n_points_LS_SHFW} points ...")
        HR_SHFW_coarse_imaging_conditions.set_imaging_conditions()
        wds_HR_SHFW, iqs_HR_SHFW = self.coarse_search(
            bounds, n_points=self.n_points_LS_SHFW
        )
        CI_HR_SHFW = self.confidence_index(iqs_HR_SHFW)
        print(f"  Confidence index (small HFW) = {CI_HR_SHFW:.3f}")

        relative_quality_HR_LHFW = CI_HR_LHFW * factor_HR_LHFW
        relative_quality_HR_SHFW = CI_HR_SHFW * factor_HR_SHFW

        if CI_HR_LHFW < alpha_LR_LHFW and CI_HR_SHFW < alpha_LR_SHFW:
            print("  Low quality image - switching to coarse search.")

            # Stage 2: quick large-HFW coarse search
            print(f"\n[Stage 2] Coarse scan (low res, large HFW = {self.hfw * 1e3:.3f} mm) "
                  f"over {self.n_points_LR_LHFW} points ...")
            LR_LHFW_coarse_imaging_conditions.set_imaging_conditions()
            wds_LR_LHFW, iqs_LR_LHFW = self.coarse_search(
                bounds, n_points=self.n_points_LR_LHFW
            )
            CI_LR_LHFW = self.confidence_index(iqs_LR_LHFW)
            print(f"  Confidence index = {CI_LR_LHFW:.3f} (threshold = {alpha_LR_LHFW:.3f})")

            if CI_LR_LHFW > alpha_LR_LHFW:
                print("  CI sufficient -> proceeding to convergent search.")
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
            print(f"\n[Stage 2] Coarse scan (low res, small HFW = {self.hfw / 50 * 1e3:.3f} mm) "
                  f"over {self.n_points_LR_SHFW} points ...")
            LR_SHFW_coarse_imaging_conditions.set_imaging_conditions()
            wds_LR_SHFW, iqs_LR_SHFW = self.coarse_search(
                bounds, n_points=self.n_points_LR_SHFW
            )
            CI_LR_SHFW = self.confidence_index(iqs_LR_SHFW)
            print(f"  Confidence index = {CI_LR_SHFW:.3f} (threshold = {alpha_LR_SHFW:.3f})")

            if CI_LR_SHFW > alpha_LR_SHFW:
                print("  CI sufficient -> proceeding to convergent search.")
                convergent_bounds = self.refine_bounds_from_coarse(
                    wds_LR_SHFW, iqs_LR_SHFW, bounds
                )
                LR_SHFW_convergent_imaging_conditions.set_imaging_conditions()
                return self.convergent_search(
                    convergent_bounds,
                    max_iterations=self.max_iterations,
                    tolerance=self.tolerance,
                )

        else:
            if relative_quality_HR_LHFW > relative_quality_HR_SHFW:
                print("  Large HFW has better signal -> using it for convergent search.")
                convergent_bounds = self.refine_bounds_from_coarse(
                    wds_HR_LHFW, iqs_HR_LHFW, bounds
                )
                HR_LHFW_convergent_imaging_conditions.set_imaging_conditions()
            else:
                print("  Small HFW has better signal -> using it for convergent search.")
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
        if self.testing:
            self._testing_wds.clear()
            self._testing_iqs.clear()
        print(f"Starting autofocus search in "
              f"[{bounds[0]*1e3:.3f}, {bounds[1]*1e3:.3f}] mm ...")
        t_total_start = time.perf_counter()
        wd = self.optimize_wd(bounds)
        if not self.simulating:
            print(f"\nSetting WD to {wd*1e3:.4f} mm on microscope ...")
            self.set_wd(wd)
        print(f"\nAutofocus finished in {time.perf_counter()-t_total_start:.3f} s.  "
              f"Optimal WD = {wd*1e3:.4f} mm")
        if self.testing:
            self._plot_testing(wd, bounds)
        return wd

    def _plot_testing(self, optimal_wd, bounds):
        """Plot all sampled WD vs sharpness metric (only called when testing=True)."""
        wds_mm = np.array(self._testing_wds) * 1e3
        iqs = np.array(self._testing_iqs)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.scatter(wds_mm, iqs, zorder=5, label="Sampled points", color="steelblue")
        ax.axvline(
            optimal_wd * 1e3,
            color="red",
            linestyle="--",
            label=f"Optimal WD = {optimal_wd*1e3:.4f} mm",
        )
        ax.axvspan(bounds[0] * 1e3, bounds[1] * 1e3, alpha=0.08, color="gray",
                   label="Search range")
        ax.set_xlabel("Working Distance (mm)")
        ax.set_ylabel("Sharpness Metric (Sobel Variance)")
        ax.set_title("Autofocus: WD vs Sharpness Metric")
        ax.legend()
        fig.tight_layout()
        plt.show()

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
        return 1/power[radius > freq_thresh].sum()
