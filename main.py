from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autofocus import Autofocus

if __name__ == "__main__":
    # ── Settings ─────────────────────────────────────────────────────────
    bounds = (0.0037, 0.0042)  # WD search range in metres (e.g. 6 mm to 10 mm)
    beam = "electron"  # "electron" or "ion"
    res = 768  # scan resolution: 768, 1536, 3072, or 6144
    hfw = 85e-6  # coarse-search horizontal field width (m)
    dwell = 1e-6  # coarse-search dwell time (s)
    tolerance = 1e-6  # convergence tolerance (m)
    testing = True  # set True to plot WD vs sharpness metric after the run

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
        hfw=hfw,
        dwell=dwell,
        simulating=False,
        tolerance=tolerance,
        testing=testing,
    )

    print(f"Searching WD in [{bounds[0]*1e3:.3f}, {bounds[1]*1e3:.3f}] mm ...")
    optimal_wd = af.find_optimal_wd(bounds)
    print(f"Autofocus complete.  Optimal WD: {optimal_wd * 1e3:.3f} mm")
