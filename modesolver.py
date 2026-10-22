"""
Minimal finite-difference waveguide mode solver (scalar / quasi-TE approximation).

Solves  (d2/dx2 + d2/dy2 + k0^2 n(x,y)^2) E = beta^2 E  on a uniform grid with
E = 0 on the boundary (domain must be large enough for the mode to decay).

Scalar approximation: good for mode SIZE and overlap trends in low/medium index
contrast (SiN/SiO2). It is not a full-vector solver; numbers should be cross-checked
against Lumerical MODE / femwell / Tidy3D before being used for a design decision.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def rect_index(x, y, core_w, core_h, n_core, n_clad):
    """Index map of a rectangular core centred at (0, 0) in a uniform cladding.

    Uses sub-pixel averaging of n^2 (area fill fraction of each cell), so that widths
    smaller than one grid step still change the result smoothly.
    """
    dx, dy = x[1] - x[0], y[1] - y[0]

    def frac(c, d, half):  # 1-D overlap of cell [c-d/2, c+d/2] with [-half, half]
        return np.clip(np.minimum(c + d / 2, half) - np.maximum(c - d / 2, -half), 0, None) / d

    f = np.outer(frac(x, dx, core_w / 2), frac(y, dy, core_h / 2))
    return np.sqrt(n_clad**2 + f * (n_core**2 - n_clad**2))


def solve_modes(n, dx, dy, wavelength, num_modes=1):
    """Return (neff array, list of normalised fields) for the index map n[ix, iy]."""
    nx, ny = n.shape
    k0 = 2 * np.pi / wavelength
    ex = np.ones(nx) / dx**2
    ey = np.ones(ny) / dy**2
    Dxx = sp.diags([ex[:-1], -2 * ex, ex[:-1]], [-1, 0, 1])
    Dyy = sp.diags([ey[:-1], -2 * ey, ey[:-1]], [-1, 0, 1])
    L = sp.kron(Dxx, sp.eye(ny)) + sp.kron(sp.eye(nx), Dyy)
    A = (L + sp.diags((k0 * n).ravel() ** 2)).tocsc()
    sigma = (k0 * n.max()) ** 2
    vals, vecs = spla.eigsh(A, k=num_modes, sigma=sigma, which="LM")
    order = np.argsort(vals)[::-1]
    neff = np.sqrt(vals[order]) / k0
    fields = []
    for i in order:
        E = vecs[:, i].reshape(nx, ny)
        E = E / np.sqrt(np.sum(np.abs(E) ** 2) * dx * dy)
        if E[nx // 2, ny // 2] < 0:
            E = -E
        fields.append(E)
    return neff, fields


def gaussian(x, y, mfd, x0=0.0, y0=0.0):
    """Fundamental fibre mode approximated as a Gaussian with mode-field diameter mfd."""
    w = mfd / 2
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / w**2)


def overlap(E1, E2, dx, dy):
    """Power coupling efficiency between two fields (butt coupling, no reflection)."""
    num = np.abs(np.sum(E1 * np.conj(E2)) * dx * dy) ** 2
    den = np.sum(np.abs(E1) ** 2) * dx * dy * np.sum(np.abs(E2) ** 2) * dx * dy
    return float(num / den)


def propagate(E, dx, dy, wavelength, n_gap, z):
    """Angular-spectrum propagation of field E over a gap z in a medium of index n_gap."""
    nx, ny = E.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k = 2 * np.pi * n_gap / wavelength
    kz = np.sqrt((k**2 - KX**2 - KY**2).astype(complex))
    return np.fft.ifft2(np.fft.fft2(E) * np.exp(1j * kz * z))


def mfd_1e2(E, x, y):
    """1/e^2 intensity mode-field diameters (MFD_x, MFD_y) from second moments."""
    I = np.abs(E) ** 2
    X, Y = np.meshgrid(x, y, indexing="ij")
    tot = I.sum()
    sx = np.sqrt((I * X**2).sum() / tot)
    sy = np.sqrt((I * Y**2).sum() / tot)
    return 4 * sx, 4 * sy  # Gaussian: MFD = 2w = 4*sigma_intensity


# ---------------------------------------------------------------- validation
def _slab_te_analytic(d, n1, n2, wl):
    """Exact TE0 neff of a symmetric slab, by bisection on the dispersion relation."""
    from scipy.optimize import brentq

    k0 = 2 * np.pi / wl
    V = k0 * d / 2 * np.sqrt(n1**2 - n2**2)
    # even TE mode: u tan(u) = sqrt(V^2 - u^2), u in (0, min(V, pi/2))
    u = brentq(lambda u: u * np.tan(u) - np.sqrt(V**2 - u**2), 1e-12, min(V, np.pi / 2) - 1e-12)
    return np.sqrt(n1**2 - (2 * u / (k0 * d)) ** 2)


def validate_slab(wl=1.55, d=0.3, n1=1.98, n2=1.444):
    """Compare the FD solver with the exact slab TE0 result (scalar is exact for slab TE)."""
    dx = 0.005
    x = np.arange(-4, 4 + dx / 2, dx)
    n = np.where(np.abs(x) <= d / 2, n1, n2)
    k0 = 2 * np.pi / wl
    m = len(x)
    e = np.ones(m) / dx**2
    A = (sp.diags([e[:-1], -2 * e, e[:-1]], [-1, 0, 1]) + sp.diags((k0 * n) ** 2)).tocsc()
    val = spla.eigsh(A, k=1, sigma=(k0 * n1) ** 2, which="LM")[0][0]
    ne_fd = np.sqrt(val) / k0
    ne_ex = _slab_te_analytic(d, n1, n2, wl)
    return ne_fd, ne_ex


if __name__ == "__main__":
    fd, ex = validate_slab()
    print(f"Slab TE0 check: FD neff = {fd:.5f}, exact = {ex:.5f}, error = {abs(fd-ex):.1e}")
