"""
Run a simulated autofocus using simulated_image/test.tif.

Ideal focus in get_simulated_image is WD = 0.008 m.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autofocus import Autofocus

WD_IDEAL = 0.008
IMAGE_PATH = Path(__file__).resolve().parent / "simulated_image" / "test.tif"


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Simulated image not found: {IMAGE_PATH}")

    autofocus = Autofocus(
        microscope=None,
        simulating=True,
        image_path=str(IMAGE_PATH),
        hfw=1e-3,
        dwell=1e-6,
        n_points_LR_LHFW=9,
        n_points_LR_SHFW=9,
        n_points_LS_LHFW=17,
        n_points_LS_SHFW=17,
        max_iterations=20,
        tolerance=1e-6,
    )

    # Search window around the known ideal focus
    bounds = (WD_IDEAL - 0.003, WD_IDEAL + 0.003)  # 5–11 mm

    # Coarse metric curve for visualization
    wds, iqs = autofocus.coarse_search(bounds, n_points=21)
    best_coarse_wd = wds[np.argmax(iqs)]
    print(f"Ideal WD:          {WD_IDEAL * 1e3:.3f} mm")
    print(f"Coarse best WD:    {best_coarse_wd * 1e3:.3f} mm")
    print(f"Coarse CI:         {autofocus.confidence_index(iqs):.3f}")

    # Full staged optimizer
    optimal_wd = autofocus.optimize_wd(bounds)
    error_mm = abs(optimal_wd - WD_IDEAL) * 1e3
    print(f"Optimized WD:      {optimal_wd * 1e3:.3f} mm")
    print(f"Absolute error:    {error_mm:.3f} mm")

    # Plot sharpness vs WD and mark results
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wds * 1e3, iqs, "o-", label="Sobel variance")
    ax.axvline(WD_IDEAL * 1e3, color="green", linestyle="--", label="Ideal WD")
    ax.axvline(optimal_wd * 1e3, color="red", linestyle="--", label="Optimized WD")
    ax.set_xlabel("Working distance (mm)")
    ax.set_ylabel("Image quality metric")
    ax.set_title("Simulated autofocus")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_plot = Path(__file__).resolve().parent / "sim_test_result.png"
    fig.savefig(out_plot, dpi=150)
    print(f"Saved plot: {out_plot}")
    # Uncomment to pop up an interactive window:
    # plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
