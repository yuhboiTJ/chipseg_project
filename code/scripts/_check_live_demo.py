"""quick: compile-check live_demo.py and show calibration values."""
import sys
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

py_compile.compile("scripts/live_demo.py", doraise=True)
print("live_demo.py compiles cleanly")

import importlib.util
spec = importlib.util.spec_from_file_location("ld", "scripts/live_demo.py")
ld = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ld)

print(f"DEFAULT_LAB_PX_PER_MM = {ld.DEFAULT_LAB_PX_PER_MM}")
print(f"lab     mm^2/pixel  = {ld.lab_mm2_per_pixel(68.6):.6e}")
print(f"derived mm^2/pixel  = {ld.derived_mm2_per_pixel('outputs/metrics.json'):.6e}")

ratio = ld.lab_mm2_per_pixel(68.6) / ld.derived_mm2_per_pixel("outputs/metrics.json")
print(f"lab / derived ratio = {ratio:.4f}")

zooms = ld.make_zoom_table(ld.lab_mm2_per_pixel(68.6))
for key, v in zooms.items():
    print(f"  zoom key {chr(key)}: {v}")
