"""
Layout of a test-chip design of experiments (DOE) for SiN inverse-taper edge couplers.

What the chip measures (each row is a facet-to-facet waveguide with a taper at each end):
  A) Tip-width sweep, 3 repeats each        -> coupling loss vs tip width (compare with simulation)
  B) Taper-length sweep at nominal tip      -> adiabaticity: is the taper long enough?
  C) Designed lateral tip offsets vs marks  -> calibrates the misalignment-tolerance curve
                                               used for the assembly-yield estimate
  Rows on a 127 um pitch (standard fibre-array pitch) so a fibre array can probe several at once.

Writes:  layout/doe_chip.gds  (needs `pip install gdstk`)
         layout/doe_chip_preview.png and layout/doe_table.csv (always)
Layers:  1/0 SiN core, 10/0 text labels, 11/0 alignment marks, 99/0 chip outline / dicing facet.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

CHIP_L = 3000.0       # um, facet to facet
W_WG = 1.0            # um, SiN routing waveguide width (single mode for ~300 nm SiN at 1550 nm)
PITCH = 127.0         # um, fibre-array pitch
L_TAPER_NOM = 300.0   # um
TIP_NOM = 0.25        # um, chosen from coupler_study.py
L_SiN, L_TXT, L_MARK, L_CHIP = 1, 10, 11, 99

polys = []   # (layer, [(x, y), ...])
labels = []  # (layer, text, (x, y), size)
table = []


def taper_poly(x0, x1, w0, w1, yc):
    return [(x0, yc - w0 / 2), (x1, yc - w1 / 2), (x1, yc + w1 / 2), (x0, yc + w0 / 2)]


def add_row(i, group, tip, l_taper, offset=0.0):
    """Facet-to-facet waveguide; the tip at x=0 is shifted laterally by `offset` from the row centre."""
    yc = i * PITCH
    # left: taper with tip at the facet (optionally offset), S-free linear lateral shift inside the taper
    polys.append((L_SiN, [(0, yc + offset - tip / 2), (l_taper, yc - W_WG / 2),
                          (l_taper, yc + W_WG / 2), (0, yc + offset + tip / 2)]))
    polys.append((L_SiN, [(l_taper, yc - W_WG / 2), (CHIP_L - l_taper, yc - W_WG / 2),
                          (CHIP_L - l_taper, yc + W_WG / 2), (l_taper, yc + W_WG / 2)]))
    polys.append((L_SiN, taper_poly(CHIP_L - l_taper, CHIP_L, W_WG, tip, yc)))
    name = f"{group}{i:02d}_t{tip*1e3:.0f}_L{l_taper:.0f}" + (f"_o{offset*1e3:+.0f}" if offset else "")
    labels.append((L_TXT, name, (l_taper + 50, yc + 8), 10))
    # alignment mark: 2 um square on the nominal row axis, 20 um from the facet
    polys.append((L_MARK, [(20, yc + 30), (22, yc + 30), (22, yc + 32), (20, yc + 32)]))
    table.append({"row": i, "group": group, "y_um": yc, "tip_width_um": tip,
                  "taper_length_um": l_taper, "lateral_offset_um": offset, "name": name})


i = 0
for tip in [0.17, 0.20, 0.23, 0.25, 0.28, 0.32, 0.40]:   # A) tip sweep, 3 repeats
    for _ in range(3):
        add_row(i, "A", tip, L_TAPER_NOM)
        i += 1
for lt in [50, 100, 200, 400, 600]:                        # B) taper length
    add_row(i, "B", TIP_NOM, lt)
    i += 1
for off in [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]:           # C) designed lateral offsets
    add_row(i, "C", TIP_NOM, L_TAPER_NOM, off)
    i += 1

chip_h = i * PITCH
polys.append((L_CHIP, [(0, -PITCH / 2), (CHIP_L, -PITCH / 2), (CHIP_L, chip_h - PITCH / 2), (0, chip_h - PITCH / 2)]))
labels.append((L_TXT, "SiN edge-coupler DOE v1", (CHIP_L / 2 - 400, chip_h - PITCH / 2 + 30), 40))

os.makedirs("layout", exist_ok=True)
with open("layout/doe_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
    w.writeheader()
    w.writerows(table)

try:
    import gdstk

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    top = lib.new_cell("SIN_EC_DOE_V1")
    for layer, pts in polys:
        top.add(gdstk.Polygon(pts, layer=layer, datatype=0))
    for layer, txt, pos, size in labels:
        top.add(*gdstk.text(txt, size, pos, layer=layer, datatype=0))
    lib.write_gds("layout/doe_chip.gds")
    print("wrote layout/doe_chip.gds")
except ImportError:
    print("gdstk not installed -> skipped GDS (pip install gdstk); preview and table still written")

# preview: full chip + zoom on one tip
colors = {L_SiN: "tab:blue", L_MARK: "tab:red", L_CHIP: "none"}
fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})
for a in ax:
    for layer, pts in polys:
        a.add_patch(MplPolygon(pts, closed=True, fc=colors[layer], ec="k" if layer == L_CHIP else colors[layer], lw=0.4))
ax[0].set(xlim=(-100, CHIP_L + 100), ylim=(-PITCH, chip_h), title=f"Full chip: {i} rows, {CHIP_L/1000:.0f} mm long", xlabel="x (um)", ylabel="y (um)")
for layer, txt, pos, size in labels:
    ax[0].text(*pos, txt, fontsize=4)
r = next(t for t in table if t["group"] == "C" and t["lateral_offset_um"] == 1.0)
ax[1].set(xlim=(-2, 40), ylim=(r["y_um"] - 3, r["y_um"] + 3), title=f"Zoom: {r['name']} (tip at facet)", xlabel="x (um)")
ax[1].axhline(r["y_um"], color="gray", ls=":", lw=0.8)
fig.tight_layout()
fig.savefig("layout/doe_chip_preview.png", dpi=150)
print(f"{i} rows -> layout/doe_table.csv, layout/doe_chip_preview.png")
