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

    def get_image(self, wd):
        """
        Get an image from the microscope

        """
    
    class ImagingConditions:
        def __init__(self, res, hfw, dwell):
            self.res = res
            self.hfw = hfw
            self.dwell = dwell

        def set_imaging_conditions(self, res, hfw, dwell):
            """
            Set the imaging conditions on the microscope
            """

    def get_image(self, wd, simulating=self.simulating):
        if simulating:
            image = self.get_simulated_image(wd)
        else:
            image = self.get_image(wd)

    def get_metric(self, wd, simulating=self.simulating):
        image = self.get_image(wd, simulating=self.simulating)
        sharpness = self.sobel_variance(image)
        return sharpness



    ######################### Dumb Search Algorithms #########################

    def course_search(self, bounds, n_points):
        wds = np.linspace(bounds[0], bounds[1], n_points)
        IQs = []
        for wd in wds:
            IQ = self.get_metric(wd, simulating=self.simulating)
            IQs.append(IQ)
        return IQs

    def convergent_search(self, bounds, max_iterations=15, tolerance=0.001):
        wd = average(bounds)
        optimal_wd = np.minimize_scalar(-self.get_metric, bounds=bounds, method='bounded')
        return optimal_wd.x

    def quadratic_fit(self, bounds, n_points=4):
        wds = np.linspace(bounds[0], bounds[1], n_points)
        IQs = []
        for wd in wds:
            IQ = self.get_metric(wd, simulating=self.simulating)
            IQs.append(IQ)
        fit = np.polyfit(wds, IQs, 2)
        max_IQ = -fit[1] / (2 * fit[0])
        return max_IQ


    ######################### Algorithm Logic #########################

    def optimize_wd(self, bounds):
        alpha_LR_LHFW = 1.25
        alpha_LR_SHFW = 1.25
        factor_HR_LHFW = 1.25
        factor_HR_SHFW = 1.25

        LR_LHFW_imaging_conditions = ImagingConditions(res=self.res, hfw=self.hfw, dwell=self.dwell)
        LR_SHFW_imaging_conditions = ImagingConditions(res=self.res, hfw=self.hfw / 50, dwell=self.dwell)
        HR_LHFW_imaging_conditions = ImagingConditions(res=self.res, hfw=self.hfw, dwell=self.dwell * 2)
        HR_SHFW_imaging_conditions = ImagingConditions(res=self.res, hfw=self.hfw / 50, dwell=self.dwell * 2)




        # Stage 1: Check if Large HFW Quick Scan Gives Good Results
        LR_LHFW_imaging_conditions.set_imaging_conditions()
        course_search_IQs_LR_LHFW = self.course_search(bounds, n_points=self.n_points_LR_LHFW)
        spacing_LR_LHFW = (bounds[1] - bounds[0]) / self.n_points_LR_LHFW
        CI_LR_LHFW = np.max(course_search_IQs_LR_LHFW) / np.mean(course_search_IQs_LR_LHFW)

        if CI_LR_LHFW < alpha_LR_LHFW: # proceed with converging if large hfw and quick scan gives good results
            convergent_bounds = (np.max(CI_LR_LHFW) - spacing_LR_LHFW, np.max(CI_LR_LHFW) + spacing_LR_LHFW)
            wd = self.convergent_search(convergent_bounds, max_iterations=self.max_iterations, tolerance=self.tolerance)
            return wd

        # Stage 2: Check if Small HFW Quick Scan Gives Good Results
        else:
            LR_SHFW_imaging_conditions.set_imaging_conditions()
            course_search_IQs = self.course_search(bounds, n_points=self.n_points_LR_SHFW)
            spacing_LR_SHFW = (bounds[1] - bounds[0]) / self.n_points_LR_SHFW
            CI_LR_SHFW = np.max(course_search_IQs) / np.mean(course_search_IQs)
            if CI_LR_SHFW < alpha_LR_SHFW:
                convergent_bounds = (np.max(CI_LR_SHFW) - spacing_LR_SHFW, np.max(CI_LR_SHFW) + spacing_LR_SHFW)
                wd = self.convergent_search(convergent_bounds, max_iterations=self.max_iterations, tolerance=self.tolerance)
                return wd

        
        # Stage 3: Longer Dwell/Higher S/N (at both small and large HFW and compare which is better)

        # get CIs for large HFW
        HR_LHFW_imaging_conditions.set_imaging_conditions()
        course_search_IQs_LHFW = self.course_search(bounds, n_points=self.n_points_LS_LHFW) # have double the points
        spacing_HR_LHFW = (bounds[1] - bounds[0]) / self.n_points_LS_LHFW
        CI_LHFW = np.max(course_search_IQs_LHFW) / np.mean(course_search_IQs_LHFW)

        # get CIs for Small HFW
        HR_SHFW_imaging_conditions.set_imaging_conditions()
        course_search_IQs_SHFW = self.course_search(bounds, n_points=self.n_points_LS_SHFW) # have double the points
        spacing_HR_SHFW = (bounds[1] - bounds[0]) / self.n_points_LS_SHFW
        CI_SHFW = np.max(course_search_IQs_SHFW) / np.mean(course_search_IQs_SHFW)

        relative_quality_LHFW = CI_LHFW * factor_CS_HR_LHFW
        relative_quality_SHFW = CI_SHFW * factor_CS_HR_SHFW

        # Convergent Search on the Better of LHFW and SHFW
        if relative_quality_LHFW > relative_quality_SHFW:
            HR_LHFW_imaging_conditions.set_imaging_conditions()
            HR_LHFW_bounds = (np.max(CI_LHFW) - spacing_HR_LHFW, np.max(CI_LHFW) + spacing_HR_LHFW)
            wd = self.convergent_search(HR_LHFW_bounds, max_iterations=self.max_iterations, tolerance=self.tolerance)
            return wd
        else:
            HR_SHFW_imaging_conditions.set_imaging_conditions()
            HR_SHFW_bounds = (np.max(CI_SHFW) - spacing_HR_SHFW, np.max(CI_SHFW) + spacing_HR_SHFW)
            wd = self.convergent_search(HR_SHFW_bounds, max_iterations=self.max_iterations, tolerance=self.tolerance)
            return wd


    ######################### Image Quality Metrics #########################

    def sobel_variance(self, image):
        sobel_x = sobel(image, axis=0)
        sobel_y = sobel(image, axis=1)
        variance = np.hypot(sobel_x, sobel_y)
        return variance

    def std(self, image):
        return np.std(image)

    def FFT_power_above_thresh(self, image, threshold=0.01):
        fft = np.fft.fft2(image)
        fft_power = np.abs(fft)**2
        fft_power_above_thresh = np.sum(fft_power > threshold)
        return fft_power_above_thresh


