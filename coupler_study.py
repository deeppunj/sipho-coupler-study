"""
SiN inverse-taper edge coupler: mode size, coupling, assembly tolerance and process window.

Questions this answers (mapped to the imec Passive Device Engineer job description):
  1. Optical design of a coupling interface: how does the taper-tip width set the mode size
     and the coupling to a fibre (SMF-28, UHNA7) and to an identical chip (chip-to-chip)?
  2. Packaging test/qualification: how much lateral misalignment and gap can a
     chip-to-chip (flip-chip / interposer) interface tolerate, and what assembly yield does
     a given placement accuracy give?
  3. Process window: how sensitive is the coupling to SiN thickness and tip-width variation
     (wafer-scale process), using a Gaussian-process surrogate for fast Monte Carlo.

Run:  python coupler_study.py      -> figures/*.png and results/*.csv
"""

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from modesolver import gaussian, mfd_1e2, overlap, propagate, rect_index, solve_modes

# ------------------------------------------------------------------ parameters
WL = 1.55            # um
N_SIN = 1.98         # stoichiometric LPCVD SiN at 1550 nm (PECVD SiN is lower, ~1.88-1.92)
N_OX = 1.444         # SiO2 cladding
T_SIN = 0.30         # um, nominal SiN thickness
DX, DY = 0.04, 0.02  # um grid
LX, LY = 16.0, 12.0  # um simulation window
FIBRES = {"SMF-28 (MFD 10.4 um)": 10.4, "UHNA7 (MFD 3.2 um)": 3.2}

x = np.arange(-LX / 2, LX / 2 + DX / 2, DX)
y = np.arange(-LY / 2, LY / 2 + DY / 2, DY)
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)


def mode(w, t=T_SIN, n_core=N_SIN):
    n = rect_index(x, y, w, t, n_core, N_OX)
    neff, (E,) = solve_modes(n, DX, DY, WL)
    return neff[0], E


def db(eta):
    return -10 * np.log10(np.maximum(eta, 1e-12))


def shift(E, sx, sy):
    """Shift a field by (sx, sy) um using Fourier interpolation (sub-pixel accurate)."""
    kx = 2 * np.pi * np.fft.fftfreq(len(x), DX)
    ky = 2 * np.pi * np.fft.fftfreq(len(y), DY)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return np.fft.ifft2(np.fft.fft2(E) * np.exp(-1j * (KX * sx + KY * sy)))


# ------------------------------------------------- 1. tip-width sweep
t0 = time.time()
widths = np.array([0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.28, 0.32, 0.40, 0.60, 1.00])
rows = []
fields = {}
for w in widths:
    ne, E = mode(w)
    fields[w] = E
    mx, my = mfd_1e2(E, x, y)
    row = {"tip_w_um": w, "neff": ne, "mfd_x_um": mx, "mfd_y_um": my}
    for name, mfd in FIBRES.items():
        row[name] = db(overlap(E, gaussian(x, y, mfd), DX, DY))
    rows.append(row)
    print(f"w={w:.2f}  neff={ne:.4f}  MFD={mx:.2f}x{my:.2f} um  "
          + "  ".join(f"{k.split()[0]}={row[k]:.2f} dB" for k in FIBRES))
print(f"sweep done in {time.time()-t0:.0f}s")

hdr = list(rows[0].keys())
np.savetxt("results/tip_sweep.csv", [[r[k] for k in hdr] for r in rows],
           delimiter=",", header=",".join(hdr), comments="", fmt="%.4f")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(widths, [r["mfd_x_um"] for r in rows], "o-", label="MFD x (horizontal)")
ax[0].plot(widths, [r["mfd_y_um"] for r in rows], "s-", label="MFD y (vertical)")
ax[0].set(xlabel="Tip width (um)", ylabel="Mode-field diameter (um)", xscale="log",
          title=f"SiN {T_SIN*1e3:.0f} nm inverse taper, TE, 1550 nm")
ax[0].legend()
for name in FIBRES:
    ax[1].plot(widths, [r[name] for r in rows], "o-", label=name)
ax[1].set(xlabel="Tip width (um)", ylabel="Coupling loss (dB)", xscale="log",
          title="Butt coupling to fibre (no Fresnel)")
ax[1].invert_yaxis()
ax[1].legend()
fig.tight_layout()
fig.savefig("figures/1_tip_sweep.png", dpi=150)

# pick the tip with the best UHNA7 match for the chip-to-chip study
w_best = min(rows, key=lambda r: r["UHNA7 (MFD 3.2 um)"])["tip_w_um"]
E0 = fields[w_best]
print(f"Chosen tip for chip-to-chip study: {w_best} um")

fig, ax = plt.subplots(figsize=(4.5, 3.6))
ax.imshow(np.abs(E0.T) ** 2, extent=[x[0], x[-1], y[0], y[-1]], origin="lower", cmap="magma")
ax.set(xlabel="x (um)", ylabel="y (um)", title=f"|E|^2, tip {w_best*1e3:.0f} nm",
       xlim=(-5, 5), ylim=(-4, 4))
fig.tight_layout()
fig.savefig("figures/2_mode_profile.png", dpi=150)

# ------------------------------------------------- 2. chip-to-chip tolerance
offs = np.linspace(-3, 3, 25)
loss_map = np.array([[db(overlap(E0, shift(E0, sx, sy).real, DX, DY)) for sy in offs] for sx in offs])

gaps = np.linspace(0, 10, 11)
gap_loss = {}
for label, ng in [("air gap (n=1.0)", 1.0), ("index-matched fill (n=1.444)", 1.444)]:
    gap_loss[label] = [db(overlap(E0, propagate(E0, DX, DY, WL, ng, g), DX, DY)) for g in gaps]

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
cs = ax[0].contourf(offs, offs, loss_map.T, levels=[0, 0.25, 0.5, 1, 1.5, 2, 3, 5, 10], cmap="viridis_r")
ax[0].contour(offs, offs, loss_map.T, levels=[1.0], colors="w", linewidths=1.5)
fig.colorbar(cs, ax=ax[0], label="Misalignment loss (dB)")
ax[0].set(xlabel="Lateral offset dx (um)", ylabel="Vertical offset dy (um)",
          title="Chip-to-chip, identical tips (white = 1 dB)", aspect="equal")
for label, l in gap_loss.items():
    ax[1].plot(gaps, l, "o-", label=label)
ax[1].set(xlabel="Facet gap (um)", ylabel="Gap loss (dB)", title="Longitudinal gap tolerance")
ax[1].invert_yaxis()
ax[1].legend()
fig.tight_layout()
fig.savefig("figures/3_chip_to_chip_tolerance.png", dpi=150)

# ------------------------------------------------- 3. assembly Monte Carlo
interp = RegularGridInterpolator((offs, offs), loss_map, bounds_error=False, fill_value=loss_map.max())
rng = np.random.default_rng(0)
spec_db = 1.0
sigmas = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5])
yields = []
for s in sigmas:
    pts = rng.normal(0, s, size=(20000, 2))
    yields.append(np.mean(interp(pts) <= spec_db) * 100)
fig, ax = plt.subplots(figsize=(4.8, 3.6))
ax.plot(sigmas, yields, "o-")
ax.set(xlabel="Placement accuracy, 1-sigma per axis (um)", ylabel=f"Yield (% with loss <= {spec_db} dB)",
       title="Assembly yield vs bonder accuracy", ylim=(0, 102))
fig.tight_layout()
fig.savefig("figures/4_assembly_yield.png", dpi=150)
for s, yv in zip(sigmas, yields):
    print(f"sigma={s:.2f} um -> yield {yv:.1f}%")

# ------------------------------------------------- 4. process window with a GP surrogate
# Design of experiments on (tip width, SiN thickness); target = coupling loss to UHNA7.
t_samples = np.array([0.27, 0.30, 0.33])
w_samples = np.array([0.17, 0.20, 0.23, 0.26, 0.30, 0.35])
Xd, yd = [], []
for t in t_samples:
    for w in w_samples:
        _, E = mode(w, t)
        Xd.append([w, t])
        yd.append(db(overlap(E, gaussian(x, y, 3.2), DX, DY)))
Xd, yd = np.array(Xd), np.array(yd)
kernel = ConstantKernel(1.0) * RBF(length_scale=[0.1, 0.05], length_scale_bounds=(1e-2, 10)) + WhiteKernel(1e-4)
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(Xd, yd)

# leave-one-out check of the surrogate
loo = []
for i in range(len(yd)):
    m = np.ones(len(yd), bool)
    m[i] = False
    g = GaussianProcessRegressor(kernel=gp.kernel_, optimizer=None, normalize_y=True).fit(Xd[m], yd[m])
    loo.append(g.predict(Xd[i:i + 1])[0] - yd[i])
print(f"GP surrogate leave-one-out RMSE = {np.sqrt(np.mean(np.square(loo))):.3f} dB")

# process variation: tip width +/-15 nm (1 sigma), thickness +/-3% (1 sigma)
w_nom = w_best
samples = np.column_stack([rng.normal(w_nom, 0.015, 5000), rng.normal(T_SIN, 0.03 * T_SIN, 5000)])
pred = gp.predict(samples)
print(f"Process MC at w={w_nom}: UHNA7 loss mean {pred.mean():.2f} dB, "
      f"P95 {np.percentile(pred, 95):.2f} dB")

WW, TT = np.meshgrid(np.linspace(0.17, 0.35, 60), np.linspace(0.27, 0.33, 40))
ZZ = gp.predict(np.column_stack([WW.ravel(), TT.ravel()])).reshape(WW.shape)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
cs = ax[0].contourf(WW * 1e3, TT * 1e3, ZZ, levels=12, cmap="viridis_r")
ax[0].scatter(Xd[:, 0] * 1e3, Xd[:, 1] * 1e3, c="w", s=12, label="solver runs (DOE)")
fig.colorbar(cs, ax=ax[0], label="Loss to UHNA7 (dB)")
ax[0].set(xlabel="Tip width (nm)", ylabel="SiN thickness (nm)", title="GP surrogate of the process window")
ax[0].legend(loc="upper right", fontsize=8)
ax[1].hist(pred, bins=50)
ax[1].axvline(np.percentile(pred, 95), color="k", ls="--", label="P95")
ax[1].set(xlabel="Loss (dB)", ylabel="Count", title="Monte Carlo over process variation")
ax[1].legend()
fig.tight_layout()
fig.savefig("figures/5_process_window.png", dpi=150)
print(f"all done in {time.time()-t0:.0f}s")
