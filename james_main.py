from james import Microscope

if __name__ == "__main__":
    # ── Settings ─────────────────────────────────────────────────────────
    hfw_m = 1e-4          # HFW during autofocus (m); smaller = higher mag = more sensitive
    dwell_s = 5e-6        # dwell time during autofocus (s)
    search_window_m = 1e-3  # ± search window around current WD (m)

    # ── Connect ───────────────────────────────────────────────────────────
    print("Connecting to microscope ...")
    scope = Microscope()
    scope.connect("localhost")
    print("Connected.")

    # ── Run autofocus ─────────────────────────────────────────────────────
    print("Starting autofocus ...")
    scope.auto_focus(
        hfw_m=hfw_m,
        dwell_s=dwell_s,
        search_window_m=search_window_m,
    )
    print(f"Autofocus complete.  WD = {scope.get_wd('electron') * 1e3:.3f} mm")
