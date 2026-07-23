"""
Helper functions used in the automated scripts (automated_grid_imaging.py, automated_grid_EBSD.py)
Author: J.D. Lamb
"""

# Standard library imports
import sys
import os
import time
import subprocess

# 3rd party libraries
import numpy as np

# Autoscript imports
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client import enumerations as enums
from autoscript_sdb_microscope_client import structures as structs


class Microscope(SdbMicroscopeClient):
    """A class that extends the SdbMicroscopeClient class to add some additional functionality.

    Attributes:
        None
    """

    def __init__(self):
        super().__init__()
        self.imaging_settings = None

    def set_coordinate_system(self, choice):
        if choice.lower() == "raw":
            self.specimen.stage.set_default_coordinate_system(
                enums.CoordinateSystem.RAW
            )
        elif choice.lower() == "specimen":
            self.specimen.stage.set_default_coordinate_system(
                enums.CoordinateSystem.SPECIMEN
            )

    def prepare_imaging(
        self,
        resolution=None,
        dwell_time: float = None,
        hfw: float = None,
        beam_current: float = None,
        beam_voltage: float = None,
        tilt_correction: bool = False,
        dynamic_focus: bool = False,
        reduced_area: tuple = None,
        bit_depth: int = 8,
    ):
        # Set resolution
        if resolution is not None:
            if type(resolution) == float:
                resolution = int(resolution)
            elif type(resolution) == str:
                resolution = int(resolution.strip())
            if resolution == 6144:
                resolution = enums.ScanningResolution.PRESET_6144X4096
            elif resolution == 3072:
                resolution = enums.ScanningResolution.PRESET_3072X2048
            elif resolution == 1536:
                resolution = enums.ScanningResolution.PRESET_1536X1024
            elif resolution == 768:
                resolution = enums.ScanningResolution.PRESET_768X512
            else:
                raise ValueError(
                    "Invalid resolution: {}. Please use one of 6144, 3072, 1536, 762 (the scope presets).".format(
                        resolution
                    )
                )
            self.beams.electron_beam.scanning.resolution.value = resolution

        # Set dwell time
        if dwell_time is not None:
            self.beams.electron_beam.scanning.dwell_time.value = dwell_time

        # Set hfw
        if hfw is not None:
            self.set_hfw(hfw, "electron")

        # Set beam current
        if beam_current is not None:
            self.beams.electron_beam.beam_current.value = beam_current

        # Set beam voltage
        if beam_voltage is not None:
            self.beams.electron_beam.high_voltage.value = beam_voltage

        # Set dynamic focus and tilt correction
        if tilt_correction:
            self.beams.electron_beam.angular_correction.tilt_correction.turn_on()
        else:
            self.beams.electron_beam.angular_correction.tilt_correction.turn_off()
        if dynamic_focus:
            self.beams.electron_beam.angular_correction.dynamic_focus.turn_on()
        else:
            self.beams.electron_beam.angular_correction.dynamic_focus.turn_off()

        if reduced_area is not None:
            self.beams.electron_beam.scanning.mode.set_reduced_area(*reduced_area)
        else:
            self.beams.electron_beam.scanning.mode.set_full_frame()

        return True

    def grab_frame(self, bit_depth=8):
        """Grabs a frame from the microscope.

        Args:
            bit_depth: int, bit depth of image grabbed

        Returns:
            The frame that was grabbed.
        """
        return self.imaging.grab_frame(structs.GrabFrameSettings(bit_depth=bit_depth))

    def set_hfw(self, hfw, beam="electron"):
        if beam.lower() == "electron":
            self.beams.electron_beam.horizontal_field_width.value = hfw
            # self.imaging.start_acquisition()
            time.sleep(0.25)
            # self.imaging.stop_acquisition()
        elif beam.lower() == "ion":
            self.beams.ion_beam.horizontal_field_width.value = hfw
            # self.imaging.start_acquisition()
            time.sleep(0.25)
            # self.imaging.stop_acquisition()

    def get_hfw(self, beam="electron"):
        if beam.lower() == "electron":
            return self.beams.electron_beam.horizontal_field_width.value
        elif beam.lower() == "ion":
            return self.beams.ion_beam.horizontal_field_width.value

    def set_wd(self, wd, beam="electron"):
        if beam.lower() == "electron":
            self.beams.electron_beam.working_distance.value = wd
        elif beam.lower() == "ion":
            self.beams.ion_beam.working_distance.value = wd

    def get_wd(self, beam="electron"):
        if beam.lower() == "electron":
            return self.beams.electron_beam.working_distance.value
        elif beam.lower() == "ion":
            return self.beams.ion_beam.working_distance.value

    def get_stage_position(self, axis=None):
        """Grabs the current position of the stage in the microscope.

        Args:
            axis: A string containing the axes to grab. Can be any combination of 'x', 'y', 'z', 't', 'r'.
                  i.e. 'xyz' will return the x, y, and z positions.

        Returns:
            A numpy array of the requested positions.
        """
        if axis is None:
            axis = "xyztr"
        elif isinstance(axis, tuple):
            axis = "".join(axis)
        lookup = {
            "x": self.specimen.stage.current_position.x,
            "y": self.specimen.stage.current_position.y,
            "z": self.specimen.stage.current_position.z,
            "t": self.specimen.stage.current_position.t,
            "r": self.specimen.stage.current_position.r,
        }
        out = []
        for character in axis:
            if character in lookup:
                out.append(lookup[character])
            else:
                raise ValueError("Invalid axis: {}".format(axis))
        return np.array(out)

    def move_stage(self, value, axis, max_iters=10, tol=None):
        """Moves the stage to a new position.

        Args:
            value: The new position to move to.
            axis: The axis to move along. Can be 'x', 'y', 'z', 't', 'r'.
            max_iters: The maximum number of iterations to try moving the stage.
            tol: The tolerance for the stage position.

        Returns: None"""
        kwargs = {axis: value}
        move_position = structs.StagePosition(**kwargs)
        current_position = self.get_stage_position(axis)[0]
        self.specimen.stage.absolute_move(move_position)
        if tol is not None:
            iters = 0
            delta = np.abs(value - current_position)
            print(f"Error on axis {axis} move #{iters}: {delta:.4e}")
            while delta > tol:
                iters += 1
                self.specimen.stage.absolute_move(move_position)
                time.sleep(0.25)
                current_position = self.get_stage_position(axis)[0]
                delta = np.abs(value - current_position)
                print(f"Error on axis {axis} move #{iters}: {delta:.4e}")
                if iters > max_iters:
                    raise RuntimeError(
                        f"Failed to move stage to correct position after 10 iterations. Current: {current_position:.4e}, Target: {value:.4e}"
                    )
            print(f"Number of iterations for {axis} axis move: {iters}")
        else:
            return True

    def set_active_view(self, detector, window, beam):
        """Sets view to the quad, detector, and beam of user's choice

        Args:
            detector: which detector to select "ETD" (secondary) or "CBS" (BSE)
            window: which imaging view to select 1 is upper left, 2 is upper right, 3 is lower left, 4 is lower right
            beam: the beam source to use 1 is electron, 2 is ion

        Returns: None"""
        self.imaging.set_active_view(window)
        self.imaging.set_active_device(beam)
        self.detector.type.value = detector

    def insertCBS(self):
        """Inserts the CBS detector into the microscope."""
        stdout, stderr = subprocess.Popen(
            'cbscontrol.exe "insert"',
            cwd="c:\scripts",
            shell=True,
            stdout=subprocess.PIPE,
        ).communicate()
        if b"Detector Inserted." not in stdout or (stderr != None):
            print("CBS detector is not inserted")
            sys.exit(2)

    def retractCBS(self):
        """Retracts the CBS detector from the microscope."""
        stdout, stderr = subprocess.Popen(
            'cbscontrol.exe "retract"',
            cwd="c:\scripts",
            shell=True,
            stdout=subprocess.PIPE,
        ).communicate()
        if b"Detector Retracted." not in stdout or (stderr != None):
            print("CBS detector is not retracted")
            sys.exit(2)

    def take_images(
        self, save_path, detectors, run_af=True, run_acb=True, image_settings={}
    ):
        """Takes an image with the given settings and saves it to the specified path.

        Args:
            save_path: The path to save the image to.
            detectors: The detectors to use for imaging.
            run_af: Whether to run autofocus before taking the image.
            run_acb: Whether to run auto contrast/brightness before taking the image.
            image_settings: The settings for the image as a dictionary of keyword entries.

        Returns: None
        """
        if type(detectors) == str:
            detectors = [detectors]
        save_path, ext = os.path.splitext(save_path)

        # Set focus
        if run_af:
            self.set_active_view("ETD", 1, 1)
            self.beams.electron_beam.scanning.resolution.value = (
                enums.ScanningResolution.PRESET_1536X1024
            )
            hfw = image_settings.get("hfw", self.get_hfw())
            self.auto_focus(hfw_m=hfw * 0.5)

        # Get images
        if "bit_depth" in image_settings:
            bit_depth = image_settings.pop("bit_depth")
        else:
            bit_depth = 8
        for det in detectors:
            self.set_active_view(det, 1, 1)
            if det == "CBS":
                self.insertCBS()
            if run_acb:
                self.beams.electron_beam.scanning.resolution.value = (
                    enums.ScanningResolution.PRESET_1536X1024
                )
                self.auto_functions.run_auto_cb()

            self.prepare_imaging(**image_settings)
            image = self.grab_frame(bit_depth)
            image.save(save_path + f"_{det}" + ext)
            if det == "CBS":
                self.retractCBS()

        self.set_active_view("ETD", 1, 1)
        self.beams.electron_beam.angular_correction.tilt_correction.turn_off()
        self.beams.electron_beam.angular_correction.dynamic_focus.turn_off()

    def auto_focus(self, hfw_m=1e-4, dwell_s=5e-6, search_window_m=1e-4):
        """Runs the autofocus routine.

        Inputs:
            search_window: The search window, in meters, for the autofocus routine. Default is 0.1 mm (0.0001 m)
        """
        self.set_active_view("ETD", 1, 1)
        self.set_hfw(hfw_m, beam="electron")
        self.auto_functions.run_auto_cb()

        def get_image_sharpness(wd):
            """Grab an image and approximate sharpness by the average/maximum magnitude of the intensity gradient."""
            # Set the WD
            self.set_wd(wd, "electron")
            # Grab the image
            image = np.copy(
                self.imaging.grab_frame(structs.GrabFrameSettings(bit_depth=8)).data
            )
            # Compute image gradients
            """
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            ax = ax.ravel()

            edges = feature.canny(image, sigma=1.5, mode="nearest")
            ax[0].imshow(edges)
            ax[0].set_title(f"Canny sigma 1.5 ({edges.mean():.2e})")

            edges = feature.canny(image, sigma=2.5, mode="nearest")
            ax[1].imshow(edges)
            ax[1].set_title(f"Canny sigma 2.5 ({edges.mean():.2e})")

            edges = filters.sobel(image)
            ax[2].imshow(edges)
            ax[2].set_title(f"Sobel ({edges.mean():.2e})")

            ax[3].imshow(image)
            ax[3].set_title(f"Raw image ({image.std():.2e})")

            for a in ax:
                a.axis("off")
            plt.tight_layout()
            plt.savefig(f"E:/James/ebsd_stitch_code/temp/{int(wd * 1e6)}.jpg")
            plt.close(fig)
            # """

            sharpness = image.std()
            return sharpness

        # Run a custom autofocus routine.
        # This autofocus routine uses image sharpness via FFTs to determine where the optimal WD should be.

        # Set the imaging conditions
        image_settings = {
            "resolution": "1536",
            "dwell_time": dwell_s,
            "hfw": hfw_m,
            "tilt_correction": False,
            "dynamic_focus": False,
            "reduced_area": (0.3, 0.3, 0.3, 0.3),
        }
        self.prepare_imaging(**image_settings)

        # Get the current WD and image sharpness
        wd0 = self.get_wd("electron")
        s0 = get_image_sharpness(wd0)
        print(f"Original ->  WD:{wd0 * 1000:.2f}, Sharpness:{s0:.3f}")

        # Optimize the WD
        bounds = (wd0 - search_window_m / 2, wd0 + search_window_m / 2)
        print(f"  -> Bounds: {bounds[0] * 1000:.2f}, {bounds[1] * 1000:.2f}")
        optimizer = SmartFocusOptimizer(
            evaluate_sharpness=get_image_sharpness, wd_bounds=bounds
        )
        optimal_wd = optimizer.optimize()
        self.set_wd(optimal_wd, "electron")

        s1 = get_image_sharpness(optimal_wd)
        print(f"Optimized -> WD:{optimal_wd * 1000:.2f}, Sharpness:{s1:.3f}")
        self.beams.electron_beam.scanning.mode.set_full_frame()
        return True


class SmartFocusOptimizer:
    def __init__(self, evaluate_sharpness, wd_bounds, noise_level="low"):
        self.evaluate = evaluate_sharpness
        self.wd_min, self.wd_max = wd_bounds
        self.n_averages = {"low": 1, "medium": 3, "high": 5}[noise_level]

    def evaluate_with_averaging(self, wd):
        """Reduce noise through averaging"""
        measurements = [self.evaluate(wd) for _ in range(self.n_averages)]
        return np.mean(measurements), np.std(measurements)

    def coarse_search(self, n_points=8, verbose=False):
        """Quick coarse search to find good starting region"""
        wds = np.linspace(self.wd_min, self.wd_max, n_points)
        sharpnesses = np.array([self.evaluate_with_averaging(wd)[0] for wd in wds])
        if verbose:
            print("WDs / Sharpnesses:")
            print(*zip(wds, sharpnesses))
        best_idx = np.argmax(sharpnesses)

        # Check if we havent found a maximum yet
        print(best_idx)
        if best_idx == len(sharpnesses) - 1:
            step_size = wds[-1] - wds[-2]
            while best_idx == len(sharpnesses) - 1:
                wd = wds[-1] + step_size
                sharpnesses = np.append(
                    sharpnesses, self.evaluate_with_averaging(wd)[0]
                )
                wds = np.append(wds, wd)
                best_idx = np.argmax(sharpnesses)
                if len(sharpnesses) > 1.5 * n_points:
                    break
        elif best_idx == 0:
            step_size = wds[-1] - wds[-2]
            while best_idx == 0:
                wd = wds[0] - step_size
                sharpnesses = np.insert(
                    sharpnesses, 0, self.evaluate_with_averaging(wd)[0]
                )
                wds = np.insert(wds, 0, wd)
                best_idx = np.argmax(sharpnesses)
                if len(sharpnesses) > 1.5 * n_points:
                    break

        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.plot(wds, sharpnesses)
        ax.set_xlabel("Working distance (m)")
        ax.set_ylabel("Image sharpness (a.u.")
        plt.show()
        # """

        # Return bracket around best point
        if best_idx == 0:
            return wds[0], wds[1]
        elif best_idx == len(wds) - 1:
            return wds[-2], wds[-1]
        else:
            return wds[best_idx - 1], wds[best_idx + 1]

    def fine_search(self, bracket, tolerance=1e-5, maxiter=15, verbose=False):
        """Fine search using Brent's method"""
        from scipy.optimize import minimize_scalar

        def neg_sharp(wd):
            return -self.evaluate_with_averaging(wd)[0]

        options = {"xatol": tolerance, "maxiter": maxiter}
        if verbose:
            options["disp"] = 3
        result = minimize_scalar(
            neg_sharp, bounds=bracket, method="bounded", options=options
        )
        return result.x

    def optimize(self, n_points=6, wd_tol_m=1e-5, maxiter=15, verbose=False):
        """Two-stage optimization"""
        # Stage 1: Coarse search
        bracket = self.coarse_search(n_points, verbose)
        print(
            f" -> Coarse search bracket: {bracket[0]*1000:.2f} mm to {bracket[1]*1000:.2f} mm"
        )

        # Stage 2: Fine search
        optimal_wd = self.fine_search(bracket, wd_tol_m, maxiter, verbose)

        return optimal_wd


if __name__ == "__main__":
    scope = Microscope()
    scope.connect("localhost")

    print(scope.get_hfw())
    scope.set_hfw(1e-3)
    print(scope.get_hfw())
    scope.set_hfw(100e-6)
    print(scope.get_hfw())

    exit()

    # Run autofocus
    print("Starting autofocus...")
    scope.auto_focus()
    print("Autofocus complete.")
