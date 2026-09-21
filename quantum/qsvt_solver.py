#!/usr/bin/env python3
"""Quantum version of the tridiagonal stages of the Stokes-Brinkman solver.

Standalone: it imports, reads and modifies nothing else in the project.  It
rebuilds in Python the rows that the C code assembles (momentum_row in
include/momentum_row.h, pressure_matrix in src/pressure_common.c), loads them
into a circuit with a block encoding for tridiagonals, inverts them with QSVT
and compares every result against the Thomas algorithm.

Three checks: one line per kind of matrix, one whole stage on a 4x4x4 grid, and
the full time step repeated over several steps.  The time step follows the C
formulas for still walls; only the six tridiagonal stages go through QSVT.

    python3 quantum/qsvt_solver.py [--points N] [--eps E] [--steps S]

Needs numpy, scipy and qiskit.  The simulation is exact and reads the whole
state vector, which a real machine would not allow, so the program also prints
the norm recovered from the success probability alone.
"""

import argparse
import math

import numpy as np
from scipy.optimize import brentq, least_squares
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UCRYGate, UnitaryGate
from qiskit.quantum_info import Operator, Statevector


# --- Physical parameters, copied from config.txt and the moving_sphere test ---

LENGTH = math.pi          # lx = ly = lz
NU = 1.0
T_END = 0.25
STEPS = 50
DT = T_END / STEPS
K_FREE = 1e30
K_SOLID = 2e-3
SPHERE_RADIUS = 0.18 * LENGTH
N_GRID = 4


# --- Solver matrices ---------------------------------------------------------
# Row t is a[t] x[t-1] + b[t] x[t] + c[t] x[t+1], as in the C code.

def spacing(n):
    return 2.0 * LENGTH / (2 * n - 1)


def gamma_from_k(k):
    beta = 1.0 + DT * NU / (2.0 * k)
    return DT * NU / (2.0 * beta)


def momentum_rows(gamma, h, normal):
    """Rows of momentum_row; first row imposed, last row imposed if normal."""
    w = -np.asarray(gamma) / h**2
    a, b, c = w.copy(), 1.0 - 2.0 * w, w.copy()
    a[0], b[0], c[0] = 0.0, 1.0, 0.0
    if normal:
        a[-1], b[-1], c[-1] = 0.0, 1.0, 0.0
    else:
        b[-1], c[-1] = 1.0 - 3.0 * w[-1], 0.0
    return a, b, c


def pressure_rows(n, h):
    """Rows of pressure_matrix: constant coefficients, Neumann at both ends."""
    w = -1.0 / h**2
    a, b, c = np.full(n, w), np.full(n, 1.0 - 2.0 * w), np.full(n, w)
    a[0], c[0] = 0.0, 2.0 * w
    b[-1], c[-1] = 1.0 - w, 0.0
    return a, b, c


def thomas(a, b, c, f):
    n = len(b)
    cp, fp = np.empty(n), np.empty(n)
    cp[0], fp[0] = c[0] / b[0], f[0] / b[0]
    for i in range(1, n):
        den = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / den
        fp[i] = (f[i] - a[i] * fp[i - 1]) / den
    x = np.empty(n)
    x[-1] = fp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = fp[i] - cp[i] * x[i + 1]
    return x


# --- Block encoding for tridiagonal matrices ---------------------------------
# A = D_b + D_a S + D_c S^T, with S a shift by one position.  Two "sel" qubits
# pick the term, one "diag" qubit loads the coefficient.  The block with all
# ancillas at zero is A / alpha, alpha = max|a| + max|b| + max|c|.

def increment(m):
    """+1 modulo 2^m, qubit 0 least significant."""
    qc = QuantumCircuit(m, name="+1")
    for i in reversed(range(m)):
        if i == 0:
            qc.x(0)
        else:
            qc.mcx(list(range(i)), i)
    return qc.to_gate()


def block_encoding(a, b, c, n_pos, axis_qubits):
    """Circuit on n_pos + 3 qubits: position, then sel (2), then diag (1)."""
    terms = [np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)]
    weights = [np.max(np.abs(t)) for t in terms]
    alpha = sum(weights)
    sel0, sel1, diag = n_pos, n_pos + 1, n_pos + 2

    # PREP loads sqrt(weight / alpha) on sel = 0, 1, 2.
    theta1 = 2 * math.asin(math.sqrt(weights[2] / alpha))
    theta0 = 2 * math.atan2(math.sqrt(weights[1]), math.sqrt(weights[0]))
    prep = QuantumCircuit(2, name="PREP")
    prep.ry(theta1, 1)
    prep.x(1)
    prep.cry(theta0, 1, 0)
    prep.x(1)

    # One rotation per (position, term), with cos(theta / 2) = coefficient / weight.
    size = 2**n_pos
    angles = np.zeros(4 * size)
    for t, (coef, wt) in enumerate(zip(terms, weights)):
        v = coef / wt if wt > 0 else np.zeros(size)
        angles[t * size:(t + 1) * size] = 2 * np.arccos(np.clip(v, -1, 1))

    inc = increment(len(axis_qubits))
    qc = QuantumCircuit(n_pos + 3, name="U_A")
    qc.append(prep.to_gate(), [sel0, sel1])
    qc.append(inc.control(2, ctrl_state=0), [sel0, sel1] + axis_qubits)
    qc.append(inc.inverse().control(2, ctrl_state=2), [sel0, sel1] + axis_qubits)
    qc.append(UCRYGate(list(angles)), [diag] + list(range(n_pos)) + [sel0, sel1])
    qc.append(prep.to_gate().inverse(), [sel0, sel1])
    return qc, alpha


def dense_grid(a, b, c, n_pos, axis_qubits):
    """The full matrix on the 2^n_pos points, used for the checks."""
    size = 2**n_pos
    stride = 2**axis_qubits[0]
    length = 2**len(axis_qubits)
    m = np.diag(np.asarray(b, float))
    for idx in range(size):
        along = (idx // stride) % length
        if along > 0:
            m[idx, idx - stride] = a[idx]
        if along < length - 1:
            m[idx, idx + stride] = c[idx]
    return m


# --- QSVT phase angles -------------------------------------------------------
# On each singular value s the encoding acts as R(s) = [[s, r], [r, -s]], and
# the sequence E(phi_d) R ... R E(phi_0), E(phi) = diag(e^{i phi}, e^{-i phi}),
# has a degree d polynomial P(s) in its top left entry.  We fit the angles so
# that Im P(s) = s_min / (2 s) on [s_min, 1], that is, an inverse.

def _reflections(s):
    r = np.sqrt(1 - s * s)
    return np.array([[s, r], [r, -s]]).transpose(2, 0, 1).astype(complex)


def _phase(phi):
    return np.diag([np.exp(1j * phi), np.exp(-1j * phi)])


def _poly_and_jacobian(phis, refl):
    d = len(phis) - 1
    e = [_phase(phi) for phi in phis]
    right = [np.broadcast_to(np.eye(2), refl.shape)]
    for j in range(1, d + 1):
        right.append(refl @ (e[j - 1] @ right[-1]))
    left = [None] * (d + 1)
    left[d] = np.broadcast_to(np.eye(2), refl.shape)
    for j in range(d - 1, -1, -1):
        left[j] = left[j + 1] @ e[j + 1] @ refl
    iz = np.diag([1j, -1j])
    jac = np.stack([(left[j] @ (iz @ e[j]) @ right[j])[:, 0, 0].imag
                    for j in range(d + 1)], axis=1)
    poly = (e[d] @ right[d])[:, 0, 0]
    return poly.imag, jac


_phase_cache = {}


def qsvt_phases(s_min, eps):
    """Angles for the inverse on [s_min, 1] with relative error below eps."""
    key = (round(s_min, 10), eps)
    if key not in _phase_cache:
        _phase_cache[key] = _fit_phases(s_min, eps)
    return _phase_cache[key]


def _fit_phases(s_min, eps):
    # Symmetric angles, phi_j = phi_{d-j}, as in Dong, Meng, Whaley and Lin:
    # half the parameters, and the fit converges in a few steps.
    check = np.linspace(s_min, 1, 2000)
    target_check = s_min / (2 * check)
    degree = 2 * int(0.6 * math.log(1 / eps) / s_min) + 1
    for _ in range(6):
        half = (degree + 1) // 2
        # Nodes denser towards s = 1 converge faster than Chebyshev ones here.
        nodes = s_min + (1 - s_min) * np.cos(np.linspace(0, np.pi / 2, 4 * (degree + 1)))
        refl = _reflections(nodes)
        target = s_min / (2 * nodes)
        last = {}

        def both(t):                 # polynomial and jacobian in one pass
            if t.tobytes() not in last:
                last.clear()
                poly, jac = _poly_and_jacobian(np.concatenate([t, t[::-1]]), refl)
                last[t.tobytes()] = (poly, jac[:, :half] + jac[:, ::-1][:, :half])
            return last[t.tobytes()]

        start = np.full(half, -np.pi / 2)
        start[0] = 0.0
        fit = least_squares(lambda t: both(t)[0] - target, start,
                            jac=lambda t: both(t)[1],
                            method="lm", xtol=1e-10, ftol=1e-10, gtol=1e-10,
                            max_nfev=100)
        phis = np.concatenate([fit.x, fit.x[::-1]])
        poly = _poly_and_jacobian(phis, _reflections(check))[0]
        error = np.max(np.abs(poly - target_check) / target_check)
        if error < eps:
            return phis, error
        degree = 2 * int(0.7 * degree) + 1
    raise RuntimeError(f"no phase angles found for s_min={s_min:.3g}, eps={eps:g}")


# --- Solving with QSVT -------------------------------------------------------
# Qubits: position, sel, diag, then flag and ctrl.  flag is 1 when the three
# ancillas are zero, ctrl runs the +phi and -phi sequences together.

def phase_block(n_pos, phi):
    qc = QuantumCircuit(n_pos + 5, name="phase")
    ancillas, flag, ctrl = [n_pos, n_pos + 1, n_pos + 2], n_pos + 3, n_pos + 4
    qc.mcx(ancillas, flag, ctrl_state=0)
    qc.rzz(2 * phi, flag, ctrl)
    qc.mcx(ancillas, flag, ctrl_state=0)
    return qc


def qsvt_circuit(be_matrix, n_pos, phis):
    """The QSVT sequence; be_matrix is the matrix of the block encoding of A."""
    u = UnitaryGate(be_matrix, label="U_A")
    qc = QuantumCircuit(n_pos + 5)
    qc.h(n_pos + 4)
    qc.compose(phase_block(n_pos, phis[0]), inplace=True)
    for k, phi in enumerate(phis[1:]):
        # A^T first: an odd QSVT polynomial on B gives (B^T)^-1, not B^-1.
        gate = u.adjoint() if k % 2 == 0 else u
        qc.append(gate, list(range(n_pos + 3)))
        qc.compose(phase_block(n_pos, phi), inplace=True)
    qc.h(n_pos + 4)
    return qc


def two_qubit_gates(qc):
    tq = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    return tq.count_ops().get("cx", 0)


class QuantumTridiagonal:
    """A x = f solved with QSVT; the circuit is built once and reused."""

    def __init__(self, a, b, c, n_pos, axis_qubits, eps):
        self.n_pos = n_pos
        self.be, self.alpha = block_encoding(a, b, c, n_pos, axis_qubits)
        size = 2**n_pos
        matrix = dense_grid(a, b, c, n_pos, axis_qubits)
        # For speed the simulation applies the matrix of the structured circuit.
        be_matrix = Operator(self.be).data
        self.be_error = np.max(np.abs(be_matrix[:size, :size] * self.alpha - matrix))
        sigma = np.linalg.svd(matrix, compute_uv=False)
        self.kappa = sigma[0] / sigma[-1]
        self.s_min = sigma[-1] / self.alpha
        self.phis, _ = qsvt_phases(self.s_min, eps)
        self.degree = len(self.phis) - 1
        self.circuit = qsvt_circuit(be_matrix, n_pos, self.phis)

    def solve(self, f):
        """x, and the norm of x from the success probability alone."""
        size = 2**self.n_pos
        norm_f = np.linalg.norm(f)
        if norm_f == 0:
            return np.zeros(size, complex), 0.0
        start = np.zeros(2**(self.n_pos + 5), complex)
        start[:size] = f / norm_f
        state = Statevector(start).evolve(self.circuit).data
        # Ancillas and flag zero, ctrl one: y = i (s_min alpha / 2) A^-1 f / |f|.
        y = state[2**(self.n_pos + 4):2**(self.n_pos + 4) + size]
        scale = 2 * norm_f / (self.s_min * self.alpha)
        return (y / 1j) * scale, scale * np.linalg.norm(y)

    def two_qubit_gates(self):
        return (self.degree * two_qubit_gates(self.be)
                + (self.degree + 1) * two_qubit_gates(phase_block(self.n_pos, 0.1)))


# --- The 4x4x4 grid ----------------------------------------------------------
# Fields in C order: index = i + n j + n^2 k, that is numpy arrays [k, j, i].

def coordinates(n, comp=None):
    """Node coordinates; comp staggers that axis by half a cell."""
    h = spacing(n)
    kk, jj, ii = np.meshgrid(*(np.arange(n),) * 3, indexing="ij")
    xyz = [ii * h, jj * h, kk * h]
    if comp is not None:
        xyz[comp] = xyz[comp] + h / 2
    return xyz


def permeability(xyz):
    """The sphere at rest in the middle, as in moving_sphere at t = 0."""
    dist2 = sum((q - LENGTH / 2)**2 for q in xyz)
    return np.where(dist2 <= SPHERE_RADIUS**2, K_SOLID, K_FREE)


def line_slices(n, axis):
    """The n^2 lines along axis, as slices of a [k, j, i] array."""
    lines = []
    for line in np.ndindex(n, n):
        sl = list(line)
        sl.insert(2 - axis, slice(None))
        lines.append(tuple(sl))
    return lines


def stage_rows(gamma, axis, comp):
    """Rows of one stage over the whole grid; comp None means pressure."""
    n = N_GRID
    h = spacing(n)
    a, b, c = (np.empty((n, n, n)) for _ in range(3))
    for sl in line_slices(n, axis):
        if comp is None:
            a[sl], b[sl], c[sl] = pressure_rows(n, h)
        else:
            a[sl], b[sl], c[sl] = momentum_rows(gamma[sl], h, normal=(comp == axis))
    return a, b, c


def thomas_stage(rows, f, axis):
    a, b, c = rows
    x = np.empty_like(f)
    for sl in line_slices(f.shape[0], axis):
        x[sl] = thomas(a[sl], b[sl], c[sl], f[sl])
    return x


_quantum_stages = {}


def quantum_stage(key, rows, axis, eps):
    """The stage as a QuantumTridiagonal on three registers, prepared once."""
    if (key, eps) not in _quantum_stages:
        m = int(math.log2(N_GRID))
        a, b, c = (r.ravel() for r in rows)
        _quantum_stages[(key, eps)] = QuantumTridiagonal(
            a, b, c, 3 * m, list(range(axis * m, (axis + 1) * m)), eps)
    return _quantum_stages[(key, eps)]


# --- The time step -----------------------------------------------------------
# Formulas from g_core (src/physics.c) and compute_div (src/pressure_common.c),
# in the case of zero velocity at the boundary.

def second_difference(field, axis, comp, h):
    """Second derivative with the boundary rows of the implicit stages."""
    n = field.shape[0]
    a, b, c = momentum_rows(np.ones(n), h, normal=(comp == axis))
    out = np.zeros_like(field)
    for sl in line_slices(n, axis):
        x = field[sl]
        row = b * x
        row[1:] += a[1:] * x[:-1]
        row[:-1] += c[:-1] * x[1:]
        out[sl] = x - row
    return out


def gradient(p, comp, h):
    """Forward difference of p along axis comp, on the nodes of comp."""
    g = np.zeros_like(p)
    ax = 2 - comp
    inner = [slice(None)] * 3
    inner[ax] = slice(0, -1)
    g[tuple(inner)] = np.diff(p, axis=ax) / h
    return g


def support(comp, n):
    """Where g_core computes g: away from the walls."""
    mask = np.zeros((n, n, n), bool)
    ranges = [slice(1, n)] * 3
    ranges[2 - comp] = slice(1, n - 1)
    mask[tuple(ranges)] = True
    return mask


def divergence_rhs(u, h):
    """-div(u) / dt at the pressure nodes, zero on the three lower faces."""
    d = np.zeros_like(u[0])
    for comp in range(3):
        ax = 2 - comp
        upper = [slice(None)] * 3
        upper[ax] = slice(1, None)
        d[tuple(upper)] += np.diff(u[comp], axis=ax) / h
    d[0, :, :] = 0.0
    d[:, 0, :] = 0.0
    d[:, :, 0] = 0.0
    return -d / DT


class Flow:
    """The state of the fluid and one time step, with a chosen solver."""

    def __init__(self, solve):
        n = N_GRID
        self.h = spacing(n)
        self.solve = solve
        zero = np.zeros((n, n, n))
        self.u = [zero.copy() for _ in range(3)]
        self.eta = [zero.copy() for _ in range(3)]
        self.zeta = [zero.copy() for _ in range(3)]
        self.p = zero.copy()
        self.p_star = zero.copy()

    def step(self, setup):
        h, solve = self.h, self.solve
        u, eta, zeta = [], [], []
        for comp in range(3):
            g = (setup.forcing[comp] - gradient(self.p_star, comp, h)
                 - setup.drag[comp] * self.u[comp]
                 + NU * (second_difference(self.eta[comp], 0, comp, h)
                         + second_difference(self.zeta[comp], 1, comp, h)
                         + second_difference(self.u[comp], 2, comp, h)))
            g = np.where(support(comp, N_GRID), g, 0.0)
            xi = self.u[comp] + setup.dt_over_beta[comp] * g
            e = self.eta[comp] + solve(("u", comp, 0), xi - self.eta[comp])
            z = self.zeta[comp] + solve(("u", comp, 1), e - self.zeta[comp])
            v = self.u[comp] + solve(("u", comp, 2), z - self.u[comp])
            eta.append(e)
            zeta.append(z)
            u.append(v)
        phi = divergence_rhs(u, h)
        for axis in range(3):
            phi = solve(("p", None, axis), phi)
        self.u, self.eta, self.zeta = u, eta, zeta
        self.p = self.p + phi
        self.p_star = self.p + phi

    def energy(self):
        return 0.5 * sum(np.sum(v * v) for v in self.u) * self.h**3


class FlowSetup:
    """What does not change in time: permeability, forcing, matrices."""

    def __init__(self):
        n = N_GRID
        self.drag, self.dt_over_beta, self.forcing, gammas = [], [], [], []
        for comp in range(3):
            xyz = coordinates(n, comp)
            k = permeability(xyz)
            beta = 1.0 + DT * NU / (2.0 * k)
            self.drag.append(NU / k)
            self.dt_over_beta.append(DT / beta)
            gammas.append(gamma_from_k(k))
        # A forcing that turns the fluid around the z axis.
        x_of_uy, y_of_ux = coordinates(n, 1)[0], coordinates(n, 0)[1]
        self.forcing = [np.sin(2 * y_of_ux), -np.sin(2 * x_of_uy), np.zeros((n, n, n))]
        self.rows = {}
        for axis in range(3):
            self.rows[("p", None, axis)] = stage_rows(None, axis, None)
            for comp in range(3):
                self.rows[("u", comp, axis)] = stage_rows(gammas[comp], axis, comp)

    def solver(self, method, eps):
        """solve(key, right-hand side), with Thomas or with QSVT."""
        def solve(key, f):
            rows = self.rows[key]
            axis = key[2]
            if key[0] == "u":
                # Imposed-value rows: the wall is still, so the value is zero.
                a, b, c = rows
                f = np.where((a == 0) & (b == 1) & (c == 0), 0.0, f)
            if method == "thomas":
                return thomas_stage(rows, f, axis)
            x, _ = quantum_stage(key, rows, axis, eps).solve(f.ravel())
            return x.real.reshape(f.shape)
        return solve


# --- The three checks --------------------------------------------------------

def report(title, x, x_ref, norm_from_p, solver):
    error = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
    norm_error = abs(norm_from_p - np.linalg.norm(x_ref)) / np.linalg.norm(x_ref)
    print(f"{title:<38}{solver.kappa:7.2f}{solver.alpha:8.2f}{solver.degree:8d}"
          f"{solver.n_pos + 5:8d}{solver.two_qubit_gates():10d}{error:12.1e}"
          f"{norm_error:12.1e}{solver.be_error:10.0e}")
    return error


def header(title):
    print(f"\n{title}")
    print(f"{'':<38}{'kappa':>7}{'alpha':>8}{'degree':>8}{'qubits':>8}{'CNOT':>10}"
          f"{'err. x':>12}{'err. norm':>12}{'err. BE':>10}")


def kappa_like_64_points(n):
    """The gamma that gives an n point line the kappa of a 64 point one."""
    def kappa(gamma, points):
        rows = momentum_rows(np.full(points, gamma), spacing(points), normal=False)
        m = int(math.log2(points))
        s = np.linalg.svd(dense_grid(*rows, m, list(range(m))), compute_uv=False)
        return s[0] / s[-1]
    target = kappa(gamma_from_k(K_FREE), 64)
    return brentq(lambda g: kappa(g, n) - target, 1e-8, 1e3)


def demo_lines(n, eps):
    m = int(math.log2(n))
    h = spacing(n)
    free = np.full(n, gamma_from_k(K_FREE))
    half = np.where(np.arange(n) < n // 2, gamma_from_k(K_FREE), gamma_from_k(K_SOLID))
    like_64 = np.full(n, kappa_like_64_points(n))
    cases = [
        ("momentum, tangential", momentum_rows(free, h, normal=False)),
        ("momentum, normal", momentum_rows(free, h, normal=True)),
        ("momentum, half in the obstacle", momentum_rows(half, h, normal=False)),
        ("momentum, kappa of 64 points", momentum_rows(like_64, h, normal=False)),
        ("pressure", pressure_rows(n, h)),
    ]
    f = np.sin(np.linspace(0.3, 2.8, n)) + 0.2
    header(f"1. Lines of {n} points, polynomial precision {eps:g}")
    errors = []
    for title, (a, b, c) in cases:
        solver = QuantumTridiagonal(a, b, c, m, list(range(m)), eps)
        x, norm = solver.solve(f)
        errors.append(report(title, x, thomas(a, b, c, f), norm, solver))
    return max(errors)


def demo_grid(setup, eps):
    n = N_GRID
    kk, jj, ii = np.meshgrid(*(np.arange(n),) * 3, indexing="ij")
    field = np.sin(ii + 1.0) * np.cos(0.7 * jj) + 0.3 * kk + 0.5

    header("2. 4x4x4 grid with the sphere: one stage per axis, 16 lines at once")
    errors = []
    for axis, name in enumerate("xyz"):
        for key, title in ((("u", 0, axis), f"u_x along {name}"),
                           (("p", None, axis), f"pressure along {name}")):
            rows = setup.rows[key]
            solver = quantum_stage(key, rows, axis, eps)
            x, norm = solver.solve(field.ravel())
            ref = thomas_stage(rows, field, axis)
            errors.append(report(title, x, ref.ravel(), norm, solver))
    return max(errors)


def demo_time(setup, steps, eps):
    print(f"\n3. Full time step on the 4x4x4 grid, {steps} steps, dt = {DT:g}")
    print("   Same formulas, same forcing: only the solver of the six stages changes.")
    print(f"{'step':>8}{'energy, Thomas':>18}{'energy, QSVT':>18}"
          f"{'diff. u':>12}{'diff. p':>12}")
    classic = Flow(setup.solver("thomas", eps))
    quantum = Flow(setup.solver("qsvt", eps))
    worst = 0.0
    for n in range(1, steps + 1):
        classic.step(setup)
        quantum.step(setup)
        du = (math.sqrt(sum(np.sum((a - b)**2) for a, b in zip(quantum.u, classic.u)))
              / math.sqrt(sum(np.sum(a * a) for a in classic.u)))
        dp = np.linalg.norm(quantum.p - classic.p) / np.linalg.norm(classic.p)
        worst = max(worst, du, dp)
        if n == 1 or n % max(1, steps // 10) == 0:
            print(f"{n:8d}{classic.energy():18.6e}{quantum.energy():18.6e}"
                  f"{du:12.1e}{dp:12.1e}")
    return worst


def main():
    parser = argparse.ArgumentParser(
        description="Solves the solver's tridiagonal systems with block encoding and QSVT.")
    parser.add_argument("--points", type=int, default=4,
                        help="points per line in the first check, a power of 2 (default 4)")
    parser.add_argument("--eps", type=float, default=1e-3,
                        help="relative error allowed on the polynomial (default 1e-3)")
    parser.add_argument("--steps", type=int, default=STEPS,
                        help=f"time steps of the third check (default {STEPS})")
    args = parser.parse_args()
    if args.points < 4 or args.points & (args.points - 1):
        parser.error("--points must be a power of 2, at least 4")
    if args.steps < 1:
        parser.error("--steps must be at least 1")

    print("err. x: relative distance from the Thomas solution.")
    print("err. norm: |x| recovered from the success probability alone.")
    print("err. BE: distance between the circuit block and A / alpha.")
    setup = FlowSetup()
    worst_solve = max(demo_lines(args.points, args.eps), demo_grid(setup, args.eps))
    worst_time = demo_time(setup, args.steps, args.eps)

    # Twelve solves per step, three of them chained for the pressure.
    ok_solve = worst_solve < args.eps
    ok_time = worst_time < 10 * args.eps
    print(f"\nWorst error of a single solve: {worst_solve:.1e}"
          f" (threshold {args.eps:g}): {'OK' if ok_solve else 'OUT OF TOLERANCE'}.")
    print(f"Worst difference in the time loop: {worst_time:.1e}"
          f" (threshold {10 * args.eps:g}): {'OK' if ok_time else 'OUT OF TOLERANCE'}.")
    return 0 if ok_solve and ok_time else 1


if __name__ == "__main__":
    raise SystemExit(main())
