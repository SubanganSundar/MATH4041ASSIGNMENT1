# =============================================================================
# MATH4041/MCSC6120 Assignment 1 -- Glucose control, part 1
# Author: Subangan Sundaralingam
#
#
# Model (scaled, Lecture 4):  f = -v - u(v+1) + G,  du/dt = -a u + b v + c f,  dv/dt = f
# Parameter vector, same order as g_simul.py:  p = [a, c, b, G]   (b and c swapped!)
# source code used: g_simul.py
# Usage:  python a1_simple.py        (takes less than a minute)
# Output: fig1.pdf, fig2.pdf (figures of the report) and results.txt (all tables)
# =============================================================================
import numpy as np
import matplotlib
matplotlib.use("Agg")                    # save figures to files, do not open windows
import matplotlib.pyplot as plt

nDof = 2
nPar = 4
LOG = open("results.txt", "w")


def log(s=""):
    """Print a line and also write it to results.txt."""
    print(s)
    LOG.write(s + "\n")


# --------------------------------------------------------------------------
# 1. Functions copied from g_simul.py
# --------------------------------------------------------------------------
# Time derivatives of the model. In: state u = (u,v), shape (nDof,); parameters p, shape (nPar,).
# Out: derivatives r, shape (nDof,).
def rhs(u, p):
    r = np.zeros((nDof,))
    f = -u[1] - u[0] * (u[1] + 1.0) + p[3]
    r[0] = -p[0] * u[0] + p[1] * f + p[2] * u[1]
    r[1] = f
    return r


# Jacobian. In: state u = (u,v), parameters p. Out: J, shape (nDof, nDof).
def Jac(u, p):
    J = np.zeros((nDof, nDof))
    dfdu0 = -u[1] - 1.0
    dfdu1 = -1.0 - u[0]
    J[0, 0] = -p[0] + p[1] * dfdu0
    J[0, 1] = p[1] * dfdu1 + p[2]
    J[1, 0] = dfdu0
    J[1, 1] = dfdu1
    return J


# Second-order Runge-Kutta step. In: state u, time step h, parameters p. Out: new state.
def RK2step(u, h, p):
    y0 = u
    f = rhs(y0, p)
    y1 = y0 + 0.5 * h * f
    f = rhs(y1, p)
    y2 = y0 + h * f
    return y2


# --------------------------------------------------------------------------
# 2. Equilibria (Question 3) and their type 
# --------------------------------------------------------------------------
def equilibria(p):
    """Equilibria by hand: v* = (-(a+b) +/- sqrt(D))/(2b), u* = (b/a) v*, D = (a+b)^2 + 4abG.
    Same formulas as g_simul.py, lines 49-56.
    In : p = [a, c, b, G].  Out: D and a list [upper, lower] of equilibria (empty if D < 0)."""
    a, b, G = p[0], p[2], p[3]
    D = (a + b) ** 2 + 4.0 * a * b * G
    if D < 0:
        return D, []
    upper = np.array([-(a + b) / (2 * a) + np.sqrt(D) / (2 * a), -(a + b) / (2 * b) + np.sqrt(D) / (2 * b)])
    lower = np.array([-(a + b) / (2 * a) - np.sqrt(D) / (2 * a), -(a + b) / (2 * b) - np.sqrt(D) / (2 * b)])
    return D, [upper, lower]


def eq_type(L):
    """Name of an equilibrium from its two eigenvalues L 'Some more examples
    of equilibria'). In: L, array of 2 (possibly complex) numbers. Out: string."""
    if abs(L[0].imag) > 1e-12:
        return "stable spiral" if L[0].real < 0 else "unstable spiral"
    if L[0].real * L[1].real < 0:
        return "saddle"
    return "stable node" if max(L.real) < 0 else "unstable node"


def G_SN(a, b):
    """Saddle-node value: no equilibria for G < G_SN ."""
    return -(a + b) ** 2 / (4 * a * b)


def G_H(a, b, c):
    """Value of G where tr J = 0 on the upper branch (stable for G > G_H).
    v_H = -a(1+a+c)/(ac+b); returns NaN if v_H is not on the upper branch."""
    vH = -a * (1 + a + c) / (a * c + b)
    if vH < -(a + b) / (2 * b):
        return np.nan
    return vH + b / a * vH * (vH + 1)


# --------------------------------------------------------------------------
# 3. Time stepping (the loop of g_simul.py)
# --------------------------------------------------------------------------
def simulate(u0, p, h=0.001, T=60.0):
    """Integrate from u0 to time T with RK2step; stop when v < -100 ('glucose too small').
    In : initial state u0 = (u0, v0), parameters p, time step h, final time T.
    Out: out, array with rows [t, u, v]; escaped, True if v < -100 was reached."""
    nStep = int(round(T / h))
    out = np.zeros((nStep + 1, 3))
    u = np.array(u0, dtype=float)
    out[0, :] = [0.0, u[0], u[1]]
    for step in range(nStep):
        u = RK2step(u, h, p)
        out[step + 1, :] = [(step + 1) * h, u[0], u[1]]
        if u[1] < -100.0:
            return out[:step + 2], True
    return out, False


def bounded(u0, p, h=0.002, T=100.0):
    """True if the orbit from u0 does not escape (controllability C1) up to time T."""
    out, escaped = simulate(u0, p, h, T)
    return not escaped


def bisection(test, lo, hi, tol):
    """Bisection on one number x: test(lo) is False, test(hi) is True.
    Returns the value of x where test switches, to within tol."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if test(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def fmt(L):
    """Write two eigenvalues as text, e.g. '-0.173 +/- 0.867i' or '-0.440, -3.091'."""
    if abs(L[0].imag) > 1e-12:
        return "%.3f +/- %.3fi" % (L[0].real, abs(L[0].imag))
    return "%.3f, %.3f" % (L[0].real, L[1].real)


def count_extrema(v):
    """Number of local maxima/minima of v(t) (to see whether v oscillates)."""
    dv = np.diff(v)
    dv = dv[np.abs(dv) > 1e-9]                     # ignore flat parts at the end
    return int(np.sum(dv[1:] * dv[:-1] < 0))


# --------------------------------------------------------------------------
# 4. Experiments
# --------------------------------------------------------------------------
a, b, c = 0.1, 1.0, 1.0                            # default parameters of g_simul.py

# Experiment 1: equilibria and eigenvalues for several G (Table 1) ------------
log("Default parameters a = %.2f, b = %.2f, c = %.2f" % (a, b, c))
log("G_SN = %.4f,  G_H = %.4f" % (G_SN(a, b), G_H(a, b, c)))
log("\nTable 1: equilibria and eigenvalues")
for G in [-3.5, -2.5, -2.0, -1.5, -0.5, 1.6]:
    p = [a, c, b, G]
    D, eqs = equilibria(p)
    if len(eqs) == 0:
        log("G = %5.2f: D = %.3f < 0, no equilibria" % (G, D))
        continue
    line = "G = %5.2f:" % G
    for name, e in zip(["upper", "lower"], eqs):
        L = np.linalg.eigvals(Jac(e, p))
        line += "  %s (%.3f, %.3f): %s, %s;" % (name, e[0], e[1], fmt(L), eq_type(L))
    log(line)

# Experiment 2: patient at baseline (u0, v0) = (0, 0) for several G (Table 2) --
log("\nTable 2: start at the baseline (0,0)")
log("%7s | %-9s | %7s | %7s | %8s | %s" % ("G", "outcome", "min v", "max v", "extrema", "C1 C2 C3"))
for G in [-3.5, -2.0, -1.5, -1.3, -1.2, -1.0, -0.5, 0.0, 1.6, 5.0, 20.0]:
    p = [a, c, b, G]
    out, escaped = simulate([0.0, 0.0], p, h=0.001, T=60.0)
    v = out[:, 2]
    n = count_extrema(v)
    C1 = not escaped
    C2 = C1 and v.min() > -1 and v.max() < 3
    C3 = C2 and n <= 1
    log("%7.2f | %-9s | %7.3f | %7.3f | %8d | %s  %s  %s" % (G, "escapes" if escaped else "bounded",
        v.min(), v.max(), n, "yes" if C1 else "no ", "yes" if C2 else "no ", "yes" if C3 else "no "))

# Experiment 3: threshold G' for the baseline patient, and how it depends on c and a (Table 3)
log("\nTable 3: thresholds (b = 1). G' = smallest G for which the baseline (0,0) stays bounded")
for (aa, cc) in [(0.1, 0.0), (0.1, 1.0), (0.1, 3.0), (0.3, 1.0)]:
    gh = G_H(aa, b, cc)
    Gprime = bisection(lambda G: bounded([0.0, 0.0], [aa, cc, b, G]), -2.5, -0.999, 1e-4)
    log("a = %.2f, c = %.1f:  G_SN = %.3f,  G_H = %.3f,  G' = %.3f" % (aa, cc, G_SN(aa, b), gh, Gprime))

# Experiment 4: a point of the boundary of the controllable set, as in the comment of
# g_simul.py ("u0 = [-11,-1.29943] close to stable manifold for p=[0.1,1,1,2]")
v_bd = bisection(lambda v0: bounded([-11.0, v0], [a, c, b, 2.0]), -2.0, -1.0, 1e-5)
log("\nBoundary point for G = 2 at u0 = -11:  v0 = %.5f   (g_simul.py comment: -1.29943)" % v_bd)

# Experiment 5: step-size check (numerical results depend on h)
o1, _ = simulate([-8.0, -0.5], [a, c, b, 1.6], h=0.001, T=30.0)
o2, _ = simulate([-8.0, -0.5], [a, c, b, 1.6], h=0.0005, T=30.0)
log("Step-size check: max difference between h = 0.001 and h = 0.0005 on [0,30]: %.1e"
    % np.max(np.abs(o1[:, 1:] - o2[::2, 1:])))

# --------------------------------------------------------------------------
# 5. Figures
# --------------------------------------------------------------------------
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8})

# Figure 1: (a) branch of equilibria versus G, (b) v(t) from the baseline
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
vv = np.linspace(-2.2, 0.8, 1500)
GG = vv + b / a * vv * (vv + 1)                    # equilibrium condition solved for G
up = vv > -(a + b) / (2 * b)                        # upper branch
st = np.array([max(np.linalg.eigvals(Jac([b / a * v, v], [a, c, b, 0.0])).real) < 0 for v in vv])
ax[0].axhspan(-2.2, -1, color="0.92")
ax[0].plot(GG[~up], vv[~up], "r--", label="lower branch: saddle")
ax[0].plot(np.where(up & ~st, GG, np.nan), vv, "r:", lw=1.5, label="upper branch: unstable")
ax[0].plot(np.where(up & st, GG, np.nan), vv, "g-", lw=1.5, label="upper branch: stable")
ax[0].plot(G_SN(a, b), -(a + b) / (2 * b), "ko", ms=4)
ax[0].text(G_SN(a, b) - 0.5, -(a + b) / (2 * b) + 0.15, "$G_{SN}$")
vH = -a * (1 + a + c) / (a * c + b)
ax[0].plot(G_H(a, b, c), vH, "ks", ms=4)
ax[0].text(G_H(a, b, c) - 0.3, vH + 0.15, "$G_H$")
ax[0].text(-3.5, -2.05, "$v<-1$: no glucose", color="0.4")
ax[0].set_xlim(-3.6, 2); ax[0].set_ylim(-2.2, 0.8)
ax[0].set_xlabel("$G$"); ax[0].set_ylabel("$v^*$")
ax[0].legend(loc="lower right", frameon=False, fontsize=7)
ax[0].set_title("(a) equilibria $v^*$ versus $G$ ($u^*=\\frac{b}{a}v^*$)")
for G, col in [(-1.3, "r"), (-1.2, "tab:orange"), (-0.5, "b"), (1.6, "g")]:
    out, escaped = simulate([0.0, 0.0], [a, c, b, G], h=0.001, T=40.0)
    ax[1].plot(out[:, 0], out[:, 2], color=col, label="$G=%.1f$%s" % (G, " (escapes)" if escaped else ""))
ax[1].axhline(-1, color="k", ls="--", lw=0.7)
ax[1].set_xlim(0, 40); ax[1].set_ylim(-2, 0.8)
ax[1].set_xlabel("$t$"); ax[1].set_ylabel("$v(t)$")
ax[1].legend(loc="lower right", fontsize=7, framealpha=1.0)
ax[1].set_title("(b) start at the baseline $(u_0,v_0)=(0,0)$")
fig.tight_layout()
fig.savefig("fig1.pdf")
plt.close(fig)

# Figure 2: phase portraits for one G in each regime (style of g_simul.py)
ics = [(-15, 3), (-9, 3), (-3, 3), (3, 3), (7, 3), (7, 0.5), (7, -0.7), (7, -2), (-15, 0),
       (-15, -0.8), (-12, -2.8), (-4, -2.8), (2, -2.8), (-6, 0.5), (-1, -0.5), (0, 0)]
cases = [(-3.5, "(a) $G=-3.5$: no equilibria"), (-2.0, "(b) $G=-2.0$: upper equilibrium unstable"),
         (-1.5, "(c) $G=-1.5$: stable spiral and saddle"), (1.6, "(d) $G=1.6$ (default of g_simul.py)")]
boundary_u0 = {-1.5: [-14, -10, -7, -4, -1.5, 0.0], 1.6: [-15, -11, -7]}
fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True, sharey=True)
for A, (G, title) in zip(axs.ravel(), cases):
    p = [a, c, b, G]
    for u0 in ics:                                         # orbits: green = bounded, red = escapes
        out, escaped = simulate(u0, p, h=0.001, T=40.0)
        A.plot(out[:, 1], out[:, 2], color="r" if escaped else "g", lw=0.7)
        A.plot(u0[0], u0[1], "k.", ms=3)
    D, eqs = equilibria(p)
    for e in eqs:                                          # equilibria and eigenvectors (as g_simul.py)
        L, V = np.linalg.eig(Jac(e, p))
        for j in range(2):
            if abs(L[j].imag) < 1e-12:
                w = V[:, j].real / np.hypot(V[0, j].real / 25, V[1, j].real / 6) * 0.08
                A.plot([e[0] - w[0], e[0] + w[0]], [e[1] - w[1], e[1] + w[1]],
                       "g" if L[j].real < 0 else "r", lw=1.5)
        A.plot(e[0], e[1], "ko", ms=4, mfc="k" if max(L.real) < 0 else "w")
    if G in boundary_u0:                                   # boundary of the controllable set by bisection
        lo = -0.999 if G < -1 else -2.9
        hi = 3.0 if G < -1 else -1.0
        pts = [(u0, bisection(lambda v0: bounded([u0, v0], p), lo, hi, 1e-3)) for u0 in boundary_u0[G]]
        A.plot([q[0] for q in pts], [q[1] for q in pts], "bD--", ms=3.5, lw=0.8,
               label="boundary (bisection)")
        A.legend(loc="upper right", fontsize=6.5, framealpha=0.9)
        log("Boundary points for G = %.1f: %s" % (G, ", ".join("(%.1f, %.3f)" % q for q in pts)))
    A.axhline(-1, color="k", ls="--", lw=0.7)
    A.set_xlim(-16, 8); A.set_ylim(-3, 3.3)
    A.set_title(title)
for A in axs[1]:
    A.set_xlabel("$u$")
for A in axs[:, 0]:
    A.set_ylabel("$v$")
fig.tight_layout()
fig.savefig("fig2.pdf")
plt.close(fig)

LOG.close()
print("\nDone: fig1.pdf, fig2.pdf and results.txt written.")
