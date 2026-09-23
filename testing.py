import time

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autofocusv2 import Autofocus

if __name__ == "__main__":
    # ── Settings ─────────────────────────────────────────────────────────
    bounds = (0.013, 0.014)  # WD search range in metres (e.g. 6 mm to 10 mm)
    beam = "electron"  # "electron" or "ion"
    res = 768  # scan resolution: 768, 1536, 3072, or 6144
    hfw_large = 10e-6  # coarse-search horizontal field width (m)
    hfw_small = 20e-6  # fine-search horizontal field width (m)
    dwell = 1e-6  # coarse-search dwell time (s)
    tolerance = 1e-6  # convergence tolerance (m)
    testing = True  # set True to plot WD vs sharpness metric after the run
    gaussian_fit = (
        False  # set True to fit a Gaussian to the sharpness metric vs WD data
    )

    # ── Connect ───────────────────────────────────────────────────────────
    print("Connecting to microscope ...")
    microscope = SdbMicroscopeClient()
    microscope.connect("localhost")
    print("Connected.")

    # ── Run autofocus ─────────────────────────────────────────────────────
    af = Autofocus(
        microscope=microscope,
        beam=beam,
        res=res,
        hfw_large=hfw_large,
        hfw_small=hfw_small,
        dwell=dwell,
        simulating=False,
        tolerance=tolerance,
        testing=testing,
        do_gaussian_fit=gaussian_fit,
    )

    # af.set_wd(0.003)
    # af.set_wd(0.004)
    # af.set_wd(0.005)
    # af.set_wd(0.0138)


    # print(af.get_resolution()[0])
    # print(af.get_resolution()[1])

    # # x resolution

    # print(af.get_dwell())

    image_time = af.get_resolution()[0] * af.get_resolution()[1] * af.dwell
    print(image_time)

    af.set_wd(0.0138)
    time.sleep(image_time)
    
    af.set_wd(0.014)
    time.sleep(image_time)

    af.set_wd(0.0138)
    time.sleep(image_time)

    af.set_wd(0.014)
    time.sleep(image_time)

    af.set_wd(0.013)
    time.sleep(image_time)

    af.set_wd(0.014)
    time.sleep(image_time)

    af.set_wd(0.0138)
    time.sleep(image_time)

    af.set_wd(0.014)
    time.sleep(image_time)

    af.set_wd(0.0138)
    time.sleep(image_time)

    af.set_wd(0.014)
    time.sleep(image_time)

    af.set_wd(0.0138)
    time.sleep(image_time)
            
