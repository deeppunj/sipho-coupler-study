"""
Expanded-beam micro-optic coupling: SMF-28 fibre -> SiN edge coupler, with Optiland.

Why: coupler_study.py shows that direct chip-to-chip / fibre-to-chip butt coupling has a
1 dB lateral tolerance below 1 um, so passive placement gives poor yield. A two-lens
expanded-beam relay moves the sub-um alignment to wafer-level lens placement and leaves
a mating interface (between the two lenses) with tens of um of lateral tolerance, at the
cost of a tight angular tolerance.

Part A (NumPy, Gaussian beam theory): lens focal lengths from mode matching, lateral and
         angular tolerance at the collimated interface vs direct butt coupling.
Part B (Optiland, real ray tracing): builds the two-lens relay, checks the paraxial
         magnification, and compares the real-ray spot with the chip mode size for a
         spherical and an aspheric (conic) focusing lens.

Run:  pip install optiland     then   python microoptic_coupling.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------- Part A: Gaussian design
WL = 1.55                 # um
MFD_FIBRE = 10.4          # um, SMF-28 at 1550 nm
MFD_CHIP = 3.98           # um, 250 nm SiN tip (from coupler_study.py)
w_f, w_c = MFD_FIBRE / 2, MFD_CHIP / 2
theta_f = WL / (np.pi * w_f)          # far-field 1/e^2 half-angle (rad)
theta_c = WL / (np.pi * w_c)
m = w_c / w_f                         # required magnification fibre -> chip
F1 = 1000.0                           # um, collimating lens focal length (choice)
F2 = F1 * m                           # um, focusing lens focal length
W = F1 * theta_f                      # um, collimated beam radius at the interface
k1db = np.sqrt(np.log(10 ** 0.1))     # exp(-x^2) = -1 dB  ->  x = 0.48

print(f"Mode matching: m = {m:.3f}, f1 = {F1:.0f} um, f2 = {F2:.0f} um")
print(f"Fibre NA(1/e2) = {theta_f:.3f}, chip-side NA(1/e2) = {theta_c:.3f}")
print(f"Collimated beam radius at the interface W = {W:.0f} um")
print(f"1 dB lateral tolerance: direct {w_c*k1db:.2f} um  vs  expanded beam {W*k1db:.0f} um")
print(f"1 dB angular tolerance at the interface: {k1db*WL/(np.pi*W)*1e3:.2f} mrad")

d = np.linspace(0, 80, 400)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(d, 4.343 * (d / w_c) ** 2, label=f"direct butt coupling (w = {w_c:.1f} um)")
ax[0].plot(d, 4.343 * (d / W) ** 2, label=f"expanded beam (W = {W:.0f} um)")
ax[0].set(xlabel="Lateral offset (um)", ylabel="Loss (dB)", ylim=(0, 3), xscale="symlog",
          title="Lateral tolerance (Gaussian theory)")
ax[0].axhline(1, color="gray", ls=":")
ax[0].legend(fontsize=8)
t = np.linspace(0, 6, 200) * 1e-3
ax[1].plot(t * 1e3, 4.343 * (np.pi * W * t / WL) ** 2)
ax[1].axhline(1, color="gray", ls=":")
ax[1].set(xlabel="Tilt at the interface (mrad)", ylabel="Loss (dB)", ylim=(0, 3),
          title="Angular tolerance: the price of expansion")
fig.tight_layout()
fig.savefig("figures/6_expanded_beam_tolerance.png", dpi=150)

# ---------------------------------------------------------------- Part B: Optiland
try:
    from optiland import optic
except ImportError:
    raise SystemExit("Part B needs Optiland:  pip install optiland")

N_FS = 1.444  # fused silica at 1550 nm
try:  # an ideal constant-index material keeps the example independent of glass catalogues
    from optiland.materials import IdealMaterial

    GLASS = IdealMaterial(n=N_FS)
except Exception:  # fall back to a catalogue glass if the class name differs in your version
    GLASS = "N-BK7"
    print("IdealMaterial not found -> using N-BK7 (index differs slightly)")

T1, T2, GAP = 500.0, 200.0, 2000.0   # um: lens thicknesses, interface gap
UM = 1e-3                             # Optiland lengths are in mm


def build(conic2: float = 0.0):
    """Fibre facet (object) -> plano-convex collimator -> gap -> plano-convex focuser -> chip."""
    R1 = -(N_FS - 1) * F1             # curved side away from the fibre
    R2 = (N_FS - 1) * F2              # curved side towards the collimated beam
    lens = optic.Optic(name="Expanded-beam fibre-to-chip relay")
    lens.surfaces.add(index=0, thickness=(F1 - T1 / N_FS) * UM)                      # fibre facet
    lens.surfaces.add(index=1, radius=np.inf, thickness=T1 * UM, material=GLASS)
    lens.surfaces.add(index=2, radius=R1 * UM, thickness=GAP * UM, is_stop=True)
    lens.surfaces.add(index=3, radius=R2 * UM, thickness=T2 * UM, material=GLASS, conic=conic2)
    lens.surfaces.add(index=4, radius=np.inf, thickness=(F2 - T2 / N_FS) * UM)
    lens.surfaces.add(index=5)                                                        # chip facet
    lens.set_aperture(aperture_type="objectNA", value=1.5 * theta_f)  # 1.5x the 1/e2 cone
    lens.fields.set_type(field_type="object_height")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=WL, is_primary=True)
    lens.updater.image_solve()
    return lens


def rms_spot_um(lens):
    lens.trace(0.0, 0.0, WL, 12, "hexapolar")
    x, y = np.asarray(lens.surfaces.x[-1, :]), np.asarray(lens.surfaces.y[-1, :])
    return np.sqrt(np.mean((x - x.mean()) ** 2 + (y - y.mean()) ** 2)) / UM


for label, k in [("spherical focuser", 0.0), ("conic focuser k = -1/n^2", -1 / N_FS**2)]:
    lens = build(k)
    try:
        mag = float(lens.paraxial.magnification())
        print(f"[{label}] paraxial magnification = {mag:.3f} (target {-m:.3f})")
    except Exception as e:  # API names can move between Optiland releases
        print(f"[{label}] paraxial magnification not available: {e}")
    r = rms_spot_um(lens)
    print(f"[{label}] real-ray RMS spot = {r:.2f} um  vs chip mode radius {w_c:.2f} um"
          f"  -> {'OK' if r < 0.3 * w_c else 'aberration-limited, coupling will drop'}")

# save a labelled layout drawing of the aspheric version
lens = build(-1 / N_FS**2)
out = lens.draw(num_rays=7)
fig = out[0] if isinstance(out, tuple) else plt.gcf()
ax = fig.axes[0]


def ray_z_span(ax):
    """First and last z of the drawn rays: the fibre facet (object) and the chip facet (image)."""
    zs = [line.get_xdata() for line in ax.get_lines()]
    zs += [seg[:, 0] for col in ax.collections if hasattr(col, "get_segments") for seg in col.get_segments()]
    zs = np.concatenate([np.asarray(z, float).ravel() for z in zs if len(z)])
    return np.nanmin(zs), np.nanmax(zs)


z_fibre, z_chip = ray_z_span(ax)
d0 = (F1 - T1 / N_FS) * UM
expected = d0 + (T1 + GAP + T2 + F2 - T2 / N_FS) * UM
if abs((z_chip - z_fibre) - expected) > 0.05 * expected:  # self-check of the label positions
    print(f"WARNING: drawn length {z_chip - z_fibre:.3f} mm differs from the design "
          f"{expected:.3f} mm -> labels may be misplaced")

z_l1 = z_fibre + d0                       # lens 1 front (flat) face
z_l2 = z_l1 + (T1 + GAP) * UM             # lens 2 front (curved) face
labels = [
    (z_fibre, "SMF-28 fibre facet\n(MFD 10.4 um)", 1.25),
    (z_l1 + T1 * UM / 2, f"Lens 1: collimator\nf = {F1/1000:.1f} mm, spherical", 1.75),
    (z_l1 + (T1 + GAP / 2) * UM, f"Collimated mating interface\nbeam radius {W:.0f} um", 1.25),
    (z_l2 + T2 * UM / 2, f"Lens 2: focuser\nf = {F2/1000:.2f} mm, conic k = -1/n^2", 1.75),
    (z_chip, "SiN chip facet\n(MFD 4.0 um)", 1.25),
]
for z, text, ypos in labels:
    ax.annotate(text, xy=(z, 0), xycoords=("data", "data"), xytext=(z, ypos),
                textcoords=("data", "axes fraction"), ha="center", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
ax.set_xlabel("z (mm)")
ax.set_title("Expanded-beam relay, SMF-28 to SiN edge coupler (Optiland ray trace)", fontsize=10, pad=110)
fig.savefig("figures/7_optiland_relay_layout.png", dpi=150, bbox_inches="tight")
print("wrote figures/6_expanded_beam_tolerance.png and figures/7_optiland_relay_layout.png")
