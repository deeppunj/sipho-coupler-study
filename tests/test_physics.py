"""Physics checks: each test compares the code with a known analytic result."""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modesolver import (gaussian, mfd_1e2, overlap, propagate, rect_index,  # noqa: E402
                        solve_modes, validate_slab)

DX = DY = 0.05
x = np.arange(-15, 15 + DX / 2, DX)
y = np.arange(-15, 15 + DY / 2, DY)


def test_slab_matches_exact_solution():
    fd, exact = validate_slab()
    assert abs(fd - exact) < 1e-4


def test_gaussian_mfd_is_recovered():
    mx, my = mfd_1e2(gaussian(x, y, 4.0), x, y)
    assert abs(mx - 4.0) < 0.01 and abs(my - 4.0) < 0.01


def test_overlap_of_identical_fields_is_one():
    E = gaussian(x, y, 4.0)
    assert abs(overlap(E, E, DX, DY) - 1) < 1e-12


def test_lateral_offset_matches_gaussian_formula():
    w, d = 2.0, 1.0
    eta = overlap(gaussian(x, y, 2 * w), gaussian(x, y, 2 * w, x0=d), DX, DY)
    assert abs(eta - np.exp(-(d / w) ** 2)) < 1e-6


def test_mode_size_mismatch_matches_gaussian_formula():
    w1, w2 = 2.0, 5.2
    eta = overlap(gaussian(x, y, 2 * w1), gaussian(x, y, 2 * w2), DX, DY)
    assert abs(eta - (2 * w1 * w2 / (w1**2 + w2**2)) ** 2) < 1e-6


def test_gap_propagation_matches_gaussian_formula():
    # The analytic formula is paraxial, the code is not: use a wide (near-paraxial) beam.
    # For a 2 um waist the exact result differs by 0.25 %, a real non-paraxial effect.
    d = 0.1
    xs = np.arange(-25, 25 + d / 2, d)
    wl, n, w = 1.55, 1.444, 5.0
    zr = np.pi * n * w**2 / wl
    z = 0.7 * zr
    E = gaussian(xs, xs, 2 * w)
    Ez = propagate(E, d, d, wl, n, z)
    power_ratio = np.sum(np.abs(Ez) ** 2) / np.sum(np.abs(E) ** 2)
    assert abs(power_ratio - 1) < 1e-9
    assert abs(overlap(E, Ez, d, d) - 1 / (1 + (z / (2 * zr)) ** 2)) < 1e-3


def test_subpixel_index_changes_smoothly_with_width():
    xs, ys = np.arange(-2, 2.001, 0.04), np.arange(-2, 2.001, 0.02)
    means = [rect_index(xs, ys, w, 0.3, 1.98, 1.444).mean() for w in (0.20, 0.21, 0.22)]
    assert means[0] < means[1] < means[2]


def test_wider_core_has_higher_neff_between_bounds():
    xs, ys = np.arange(-3, 3.001, 0.05), np.arange(-2.5, 2.501, 0.025)
    neffs = []
    for w in (0.6, 1.0):
        neff, _ = solve_modes(rect_index(xs, ys, w, 0.3, 1.98, 1.444), 0.05, 0.025, 1.55)
        neffs.append(neff[0])
    assert 1.444 < neffs[0] < neffs[1] < 1.98


def test_layout_doe_runs_and_writes_32_rows(tmp_path):
    subprocess.run([sys.executable, os.path.join(ROOT, "layout_doe.py")], cwd=tmp_path, check=True)
    lines = (tmp_path / "layout" / "doe_table.csv").read_text().strip().splitlines()
    assert len(lines) == 1 + 32
