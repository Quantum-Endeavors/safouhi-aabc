#!/usr/bin/env python3
"""
Three-centre (AA|BC) two-electron Coulomb integrals over Slater-type functions by the method of

    H. Safouhi, "The S and Double-Exponential Transformations with a Moment Expansion for
    Three-Centre Coulomb and Hybrid Integrals", arXiv:2611.23965v1.

    (ab|cd) = Int Int a(r1) b(r1) (1 / |r1 - r2|) c(r2) d(r2) dr1 dr2

with a and b on centre A, c on centre B, d on centre C (three distinct centres).

Steps:
  - each normalised STF is a finite sum of B functions (paper eq. 11);
  - each B-function integral uses the Fourier-B reduction (eqs. 17, 18): finite angular sums,
    an outer Feynman integral over s in [0, 1] (Gauss-Legendre) and a semi-infinite radial
    integral K(s) containing j_lambda(v x) (eq. 19);
  - K(s) is evaluated with the S transformation (eq. 26) and the Ooura-Mori Fourier-type DE
    quadrature (phi_1, K = 6, M h = pi / v, eqs. 35-36), or with the moment expansion (eq. 68)
    when rho(s) <= 0.5 (<= 0.9 for lambda >= 3), section 7. "Direct DE" on j_lambda (eq. 40)
    is available as a cross-check (--method direct / both).

Departures from the printed formulas, needed to reproduce independent reference values:
  1. Eq. 18: Delta l = (l3' + l4' - l') / 2. The printed (l3 + l4 - l') / 2 is not an integer
     when l3' < l3.
  2. Eq. 17: the factor (-1)^((l3'+l4'+l')/2) Y_l'^(m3'-m4')(R34-hat) is replaced by
     Y_l'^(m4'-m3')(R34-hat), i.e. the Gaunt coefficient's own M and no extra sign, with
     R34 = R3 - R4. As printed, the result is wrong whenever B or C carries a p_z or d
     function, or both carry p functions.
  3. The number of Gauss-Legendre points for the s integral is an option (--outer). The
     paper's 48 points are not enough when the B and C exponents are very different (e.g. a
     tight core 1s on B); the integrand then peaks sharply near s = 0. --outer 384 is safe
     for all bundled examples.
  4. M (and hence h = pi / (v M)) is chosen per radial integral by doubling M from 2 until two
     successive values agree to 1e-14 relative, instead of being fixed per case.

Functions are normalised real STFs N r^(n-1) e^(-zeta r) Z_lm, where Z_lm are real spherical
harmonics without the Condon-Shortley phase: m > 0 ~ cos(m phi) (p: m=1 is x), m < 0 ~
sin(|m| phi) (p: m=-1 is y), m = 0 (p: z). Coordinates are in bohr, results in hartree.

Only the Python standard library is used.

Usage:
  python3 safouhi_aabc.py --A 0 0 0 --B 1.43 0 1.11 --C -1.43 0 1.11 \\
      --a 1 0 0 9.8 --b 1 0 0 9.8 --c 1 0 0 0.76 --d 1 0 0 0.76
  python3 safouhi_aabc.py --examples [--outer 384]

Each function is given as "n l m zeta".
"""

import argparse
import cmath
import functools
import math
import sys
import time

# ---------------------------------------------------------------------------------------------
# Small special functions
# ---------------------------------------------------------------------------------------------

fact = math.factorial


def dfact(n):
    """n!! for n >= -1."""
    r = 1.0
    while n > 1:
        r *= n
        n -= 2
    return r


def binom_real(a, k):
    """Generalised binomial coefficient (a choose k)."""
    r = 1.0
    for j in range(k):
        r *= (a - j) / (j + 1)
    return r


def pochhammer(a, n):
    r = 1.0
    for j in range(n):
        r *= a + j
    return r


@functools.lru_cache(maxsize=None)
def wigner3j(j1, j2, j3, m1, m2, m3):
    """Wigner 3j symbol (Racah formula), integer arguments."""
    if m1 + m2 + m3 != 0 or j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    tri = fact(j1 + j2 - j3) * fact(j1 - j2 + j3) * fact(-j1 + j2 + j3) / fact(j1 + j2 + j3 + 1)
    pre = math.sqrt(tri * fact(j1 + m1) * fact(j1 - m1) * fact(j2 + m2) * fact(j2 - m2)
                    * fact(j3 + m3) * fact(j3 - m3))
    total = 0.0
    for k in range(max(0, j2 - j3 - m1, j1 - j3 + m2), min(j1 + j2 - j3, j1 - m1, j2 + m2) + 1):
        den = (fact(k) * fact(j1 + j2 - j3 - k) * fact(j1 - m1 - k) * fact(j2 + m2 - k)
               * fact(j3 - j2 + m1 + k) * fact(j3 - j1 - m2 + k))
        total += (-1) ** k / den
    return (-1) ** (j1 - j2 - m3) * pre * total


@functools.lru_cache(maxsize=None)
def gaunt(l1, m1, l2, m2, L, M):
    """Paper eq. 9: Int conj(Y_l1^m1) Y_l2^m2 Y_L^M dOmega (Condon-Shortley Y)."""
    if m1 != m2 + M or L < abs(l1 - l2) or L > l1 + l2 or (l1 + l2 + L) % 2:
        return 0.0
    if abs(m1) > l1 or abs(m2) > l2 or abs(M) > L:
        return 0.0
    return ((-1) ** m1 * math.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * L + 1) / (4 * math.pi))
            * wigner3j(l1, l2, L, 0, 0, 0) * wigner3j(l1, l2, L, -m1, m2, M))


def plm(l, m, x):
    """Associated Legendre P_l^m (m >= 0) without the Condon-Shortley phase."""
    pmm = 1.0
    s = math.sqrt(max(0.0, (1.0 - x) * (1.0 + x)))
    for i in range(1, m + 1):
        pmm *= (2 * i - 1) * s
    if l == m:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmmp1
    pll = 0.0
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm, pmmp1 = pmmp1, pll
    return pll


def ylm(l, m, ct, phi):
    """Paper eq. 2: Y_l^m = i^(m+|m|) N P_l^|m|(cos theta) e^(i m phi)."""
    am = abs(m)
    n = math.sqrt((2 * l + 1) / (4 * math.pi) * fact(l - am) / fact(l + am))
    ph = -1.0 if (m > 0 and m % 2) else 1.0
    return ph * n * plm(l, am, ct) * cmath.exp(1j * m * phi)


def solid(l, m, v):
    """Regular solid harmonic r^l Y_l^m(r-hat), finite at r = 0."""
    r = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if r == 0.0:
        return complex(1.0 / math.sqrt(4 * math.pi)) if l == 0 else 0j
    return r ** l * ylm(l, m, v[2] / r, math.atan2(v[1], v[0]))


def sph_j(l, z):
    """Spherical Bessel j_l(z), z >= 0."""
    if z < 0.5 + l:  # power series, eq. 7
        term = z ** l / dfact(2 * l + 1)
        total = term
        for q in range(1, 200):
            term *= -z * z / (2.0 * q * (2.0 * l + 2.0 * q + 1.0))
            total += term
            if abs(term) < 1e-18 * abs(total):
                break
        return total
    j0 = math.sin(z) / z
    if l == 0:
        return j0
    j1 = math.sin(z) / (z * z) - math.cos(z) / z
    for k in range(1, l):
        j0, j1 = j1, (2 * k + 1) / z * j1 - j0
    return j1


@functools.lru_cache(maxsize=None)
def khat_poly(N):
    """Reduced Bessel k-hat_{N+1/2}(z) = e^-z sum_p c_p z^p; returns the c_p."""
    c = [0.0] * (N + 1)
    for j in range(N + 1):
        c[N - j] = fact(N + j) / (fact(j) * fact(N - j)) / 2.0 ** j
    return tuple(c)


# ---------------------------------------------------------------------------------------------
# Truncated Taylor series ("jets") in u = x^2, for (1/x d/dx)^lambda = 2^lambda d^lambda/du^lambda
# ---------------------------------------------------------------------------------------------

def jet_mul(a, b):
    n = len(a)
    c = [0.0] * n
    for i in range(n):
        for j in range(n - i):
            c[i + j] += a[i] * b[j]
    return c


def jet_pow_linear(c0, c1, p, n):
    """(c0 + c1 delta)^p to order n, c0 > 0."""
    return [binom_real(p, k) * c0 ** (p - k) * c1 ** k for k in range(n + 1)]


def jet_exp_neg(z):
    """exp(-z) for a jet z."""
    n = len(z) - 1
    e = [0.0] * (n + 1)
    e[0] = 1.0
    for k in range(1, n + 1):
        e[k] = sum(j * (-z[j]) * e[k - j] for j in range(1, k + 1)) / k
    scale = math.exp(-z[0])
    return [x * scale for x in e]


# ---------------------------------------------------------------------------------------------
# The semi-infinite radial integral K(s), eq. 19, divided by v^lambda (the v^lambda goes with
# Y_lambda^mu(v-hat) into a solid harmonic, so v = 0 needs no special case).
# Radial parameters p = (nx, nk, ngamma, nu_int, lam, zs, z3, z4, R34), nu = nu_int + 1/2.
# ---------------------------------------------------------------------------------------------

def amplitude(p, s, x, kp):
    """x^nx k-hat_nu(R34 gamma) / ((zs^2 + x^2)^nk gamma^ngamma), without the oscillatory factor."""
    nx, nk, ngamma, _, _, zs, z3, z4, R34 = p
    g2 = (1 - s) * z4 * z4 + s * z3 * z3 + s * (1 - s) * x * x
    g = math.sqrt(g2)
    z = R34 * g
    poly = 0.0
    for c in reversed(kp):
        poly = poly * z + c
    # in logs: at the far exp-sinh nodes x^nx overflows before e^-z underflows
    lg = nx * math.log(x) - z + math.log(poly) - nk * math.log(zs * zs + x * x) - ngamma * math.log(g)
    return math.exp(lg) if lg > -745 else 0.0


def g_lambda(p, s, x, kp):
    """G_{lambda,s}(x) = (1/x d/dx)^lambda [x^(nx+lambda-1) k-hat / ((zs^2+x^2)^nk gamma^ngamma)], eq. 26."""
    nx, nk, ngamma, _, L, zs, z3, z4, R34 = p
    u = x * x
    f = jet_pow_linear(u, 1.0, 0.5 * (nx + L - 1), L)                 # u^a
    f = jet_mul(f, jet_pow_linear(zs * zs + u, 1.0, -nk, L))          # (zs^2 + u)^-nk
    G0 = (1 - s) * z4 * z4 + s * z3 * z3 + s * (1 - s) * u
    g1 = s * (1 - s)
    f = jet_mul(f, jet_pow_linear(G0, g1, -0.5 * ngamma, L))          # gamma^-ngamma
    z = [R34 * c for c in jet_pow_linear(G0, g1, 0.5, L)]             # R34 gamma
    poly = [0.0] * (L + 1)
    zp = [1.0] + [0.0] * L
    for c in kp:                                                      # sum c_i z^i
        for k in range(L + 1):
            poly[k] += c * zp[k]
        zp = jet_mul(zp, z)
    f = jet_mul(f, jet_mul(poly, jet_exp_neg(z)))
    return 2.0 ** L * fact(L) * f[L]


def phi1(t, K=6.0):
    """Ooura-Mori phi_1(t) = t / (1 - exp(-K sinh t)) and its derivative."""
    if abs(t) < 1e-8:
        return 1.0 / K + 0.5 * t, 0.5
    try:
        ks = K * math.sinh(t)
        om = -math.expm1(-ks)  # 1 - E
    except OverflowError:
        return 0.0, 0.0
    if not math.isfinite(om) or abs(om) > 1e300:
        return 0.0, 0.0
    E = 1.0 - om
    return t / om, (om - t * K * math.cosh(t) * E) / (om * om)


def de_fourier(f, M, h, counter):
    """Fourier-type DE sum M h sum_k f(M phi(kh)) phi'(kh), eq. 32."""
    total = 0.0
    for direction in (+1, -1):
        small = 0
        k = 0 if direction > 0 else -1
        while abs(k) < 200000:
            ph, dph = phi1(k * h)
            # x -> 0 double exponentially; stop before x^2 underflows (the weight phi' is ~0 there)
            if ph <= 0.0 or dph == 0.0 or M * ph < 1e-100:
                break
            term = f(M * ph) * dph
            counter[0] += 1
            total += term
            scale = abs(total) if total else 1.0
            small = small + 1 if abs(term) * M * h < 1e-17 * scale else 0
            if small >= 4:
                break
            k += direction
    return M * h * total


def exp_sinh(f, counter):
    """exp-sinh quadrature on [0, inf) for a non-oscillatory integrand, refined until stable."""
    prev = 0.0
    h = 0.5
    while h > 1e-3:
        total = 0.0
        for direction in (+1, -1):
            small = 0
            k = 0 if direction > 0 else -1
            while abs(k) < 100000:
                t = k * h
                ex = 0.5 * math.pi * math.sinh(t)
                if abs(ex) > 700:
                    break
                x = math.exp(ex)
                term = f(x) * 0.5 * math.pi * math.cosh(t) * x
                counter[0] += 1
                total += term
                small = small + 1 if abs(term) < 1e-18 * abs(total) else 0
                if small >= 3:
                    break
                k += direction
        total *= h
        if h < 0.5 and abs(total - prev) <= 1e-15 * abs(total):
            return total
        prev = total
        h *= 0.5
    return prev


# radial integrals that never stabilised to 1e-14 (count, worst relative change at M = 256)
UNSTABLE = [0, 0.0]


def report_unstable():
    if UNSTABLE[0]:
        print(f"    note: {UNSTABLE[0]} radial integrals changed by up to {UNSTABLE[1]:.1e} (relative) "
              "between M = 128 and 256")
    UNSTABLE[0], UNSTABLE[1] = 0, 0.0


def radial(p, s, v, method, counter):
    """K(s) / v^lambda."""
    kp = khat_poly(p[3])
    L = p[4]
    alpha = p[8] * math.sqrt(s * (1 - s))
    rho = v / alpha
    if rho <= (0.9 if L >= 3 else 0.5):  # moment expansion, eq. 68
        total = 0.0
        for q in range(400):
            c = (-1) ** q * v ** (2 * q) / (2.0 ** q * fact(q) * dfact(2 * L + 2 * q + 1))
            pq = (p[0] + L + 2 * q,) + p[1:]
            term = c * exp_sinh(lambda x: amplitude(pq, s, x, kp), counter)
            total += term
            if abs(term) < 1e-17 * abs(total) and q > 2:
                break
        else:
            print(f"warning: moment series not converged (rho {rho:.3f})", file=sys.stderr)
        return total
    # choose M by doubling until two successive values agree
    prev = prev_prev = 0.0
    M = 2.0
    while M <= 256.0:
        if method == "sde":
            h = math.pi / (v * M)  # eq. 35
            val = de_fourier(lambda x: g_lambda(p, s, x, kp) * math.sin(v * x), M, h, counter)
            val /= v ** (2 * L + 1)  # eq. 26: v^-(lambda+1), and / v^lambda
        else:
            h = math.pi / (2 * v * M)  # eq. 40
            val = de_fourier(lambda x: amplitude(p, s, x, kp) * sph_j(L, v * x), M, h, counter)
            val /= v ** L
        if M > 2.0 and abs(val - prev) <= 1e-14 * abs(val) + 1e-300:
            return val
        prev_prev, prev = prev, val
        M *= 2.0
    UNSTABLE[0] += 1
    UNSTABLE[1] = max(UNSTABLE[1], abs(val - prev_prev) / abs(val) if val else 0.0)
    return prev


def gauss_legendre01(n):
    """n-point Gauss-Legendre nodes and weights on [0, 1]."""
    xs, ws = [], []
    for i in range(n):
        z = math.cos(math.pi * (i + 0.75) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, z
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * z * p1 - (k - 1) * p0) / k
            dp = n * (z * p1 - p0) / (z * z - 1)
            dz = p1 / dp
            z -= dz
            if abs(dz) < 1e-16:
                break
        xs.append(0.5 * (1 - z))
        ws.append(1.0 / ((1 - z * z) * dp * dp))
    return xs, ws


# ---------------------------------------------------------------------------------------------
# B-function integral, eq. 17
# ---------------------------------------------------------------------------------------------

def ipow(k):
    return (1, 1j, -1, -1j)[k % 4]


class Engine:
    def __init__(self, R3, R4, outer=48, method="sde"):
        """R3, R4: centres B and C relative to A."""
        self.R3, self.R4, self.method = R3, R4, method
        self.sx, self.sw = gauss_legendre01(outer)
        self.v = [[(1 - s) * R3[c] + s * R4[c] for c in range(3)] for s in self.sx]
        self.cache = {}
        self.nodes = [0]

    def radial_table(self, p):
        """K(s) / v^lambda at the s nodes, cached on the radial parameters."""
        if p not in self.cache:
            self.cache[p] = [radial(p, s, math.sqrt(sum(c * c for c in v)), self.method, self.nodes)
                             for s, v in zip(self.sx, self.v)]
        return self.cache[p]

    def bfun_integral(self, b1, b2, b3, b4):
        """Eq. 17 for B functions b = (n, l, m, zeta): [B1]* B2 on A, [B3]* on B, B4 on C."""
        n1, l1, m1, z1 = b1
        n2, l2, m2, z2 = b2
        n3, l3, m3, z3 = b3
        n4, l4, m4, z4 = b4
        zs = z1 + z2
        R34v = [self.R3[c] - self.R4[c] for c in range(3)]
        R34 = math.sqrt(sum(c * c for c in R34v))
        pre = (8 * (4 * math.pi) ** 3 * math.sqrt(math.pi) * z1 ** l1 * z2 ** l2
               * z3 ** (2 * n3 + l3 - 1) * z4 ** (2 * n4 + l4 - 1) * dfact(2 * l3 + 1) * dfact(2 * l4 + 1)
               * fact(n3 + l3 + n4 + l4 + 1)
               / (2.0 ** (l1 + l2 + 1) * fact(n1 + l1) * fact(n2 + l2) * fact(n3 + l3) * fact(n4 + l4)))
        total = 0j
        for l in range(abs(l1 - l2), l1 + l2 + 1):
            gl = gaunt(l2, m2, l1, m1, l, m2 - m1)
            if gl == 0.0:
                continue
            tl = (-1j) ** l * gl / 2.0 ** (2 * n1 + 2 * n2 + l)
            for k in range(2, n1 + n2 + 1):
                ck = sum(2.0 ** k * fact(2 * n1 - i - 1) * fact(2 * n2 - k + i - 1) * z1 ** (i - 1) * z2 ** (k - i - 1)
                         / (fact(i - 1) * fact(n1 - i) * fact(k - i - 1) * fact(n2 - k + i))
                         for i in range(max(1, k - n2), min(n1, k - 1) + 1))
                if ck == 0.0:
                    continue
                nk = k + l1 + l2
                eta = l - k - l1 - l2 + 1
                etap = -eta // 2 if eta % 2 == 0 else -(eta + 1) // 2
                fk = zs ** (nk - l - 1) * math.gamma(k + l1 + l2 + l + 1) / math.gamma(l + 1.5)
                for l4p in range(l4 + 1):
                    for m4p in range(-l4p, l4p + 1):
                        if abs(m4 - m4p) > l4 - l4p:
                            continue
                        g4 = gaunt(l4, m4, l4 - l4p, m4 - m4p, l4p, m4p)
                        if g4 == 0.0:
                            continue
                        f4 = ipow(l4 + l4p) * (-1) ** l4p * g4 / (dfact(2 * l4p + 1) * dfact(2 * (l4 - l4p) + 1))
                        for l3p in range(l3 + 1):
                            for m3p in range(-l3p, l3p + 1):
                                if abs(m3 - m3p) > l3 - l3p:
                                    continue
                                g3 = gaunt(l3, m3, l3 - l3p, m3 - m3p, l3p, m3p)
                                if g3 == 0.0:
                                    continue
                                f3 = ipow(l3 + l3p) * g3 / (dfact(2 * l3p + 1) * dfact(2 * (l3 - l3p) + 1))
                                for lp in range(l3p + l4p + 1):
                                    gp = gaunt(l4p, m4p, l3p, m3p, lp, m4p - m3p)
                                    if gp == 0.0:
                                        continue
                                    # correction 2: printed (-1)^((l3'+l4'+l')/2) Y_l'^(m3'-m4')(R34-hat)
                                    fp = gp * solid(lp, m4p - m3p, R34v)
                                    m34 = (m3 - m3p) - (m4 - m4p)
                                    mu = (m2 - m1) - m34
                                    # correction 1: Delta l = (l3' + l4' - l') / 2
                                    dl = (l3p + l4p - lp) // 2
                                    for l34 in range(l3 - l3p + l4 - l4p + 1):
                                        g34 = gaunt(l3 - l3p, m3 - m3p, l4 - l4p, m4 - m4p, l34, m34)
                                        if g34 == 0.0:
                                            continue
                                        for lam in range(l + l34 + 1):
                                            gla = gaunt(l, m2 - m1, l34, m34, lam, mu)
                                            if gla == 0.0:
                                                continue
                                            flam = ipow(lam) * gla
                                            ysol = [solid(lam, mu, v) for v in self.v]
                                            for j in range(dl + 1):
                                                fj = (binom_real(dl, j) * (-1) ** j
                                                      / (2.0 ** (n3 + n4 + l3 + l4 - j + 1) * fact(n3 + n4 + l3 + l4 - j + 1)))
                                                nu_int = n3 + n4 + l3 + l4 - lp - j
                                                ngamma = 2 * (n3 + l3 + n4 + l4) - (l3p + l4p) - lp + 1
                                                for r in range(etap + 1):
                                                    fr = ((-1) ** r * pochhammer(eta / 2, r) * pochhammer((eta + 1) / 2, r)
                                                          / (pochhammer(l + 1.5, r) * fact(r) * zs ** (2 * r)))
                                                    if fr == 0.0:
                                                        continue
                                                    p = (l3 - l3p + l4 - l4p + 2 * r + l, nk, ngamma, nu_int, lam,
                                                         zs, z3, z4, R34)
                                                    kv = self.radial_table(p)
                                                    sint = sum(w * s ** (n3 + l3 + l4 - l4p) * (1 - s) ** (n4 + l4 + l3 - l3p)
                                                               * y * kval
                                                               for s, w, y, kval in zip(self.sx, self.sw, ysol, kv))
                                                    total += tl * ck * fk * f4 * f3 * fp * g34 * flam * fj * fr * sint
        return pre * total

    @staticmethod
    def sto_to_b(n, l, m, zeta):
        """Normalised STF (n, l, complex m, zeta) as a sum of B functions, eq. 11."""
        norm = math.sqrt((2 * zeta) ** (2 * n + 1) / fact(2 * n)) / zeta ** (n - 1)
        out = []
        for p in range((n - l + 1) // 2, n - l + 1):
            c = ((-1) ** (n - l - p) * fact(n - l) * 2.0 ** (2 * p + 2 * l - n) * fact(l + p)
                 / (fact(2 * p - n + l) * fact(n - l - p)))
            out.append((norm * c, (p, l, m, zeta)))
        return out

    def sto_integral(self, s1, s2, s3, s4):
        """Complex-harmonic STF integral [chi1]* chi2 [chi3]* chi4."""
        total = 0j
        for c1, b1 in self.sto_to_b(*s1):
            for c2, b2 in self.sto_to_b(*s2):
                for c3, b3 in self.sto_to_b(*s3):
                    for c4, b4 in self.sto_to_b(*s4):
                        total += c1 * c2 * c3 * c4 * self.bfun_integral(b1, b2, b3, b4)
        return total


def real_to_complex(m):
    """Real harmonic (m > 0 ~ sqrt2 cos, m < 0 ~ sqrt2 sin, no Condon-Shortley phase) as (m, coefficient) pairs of Y_l^m."""
    if m == 0:
        return [(0, 1.0)]
    r2 = 1 / math.sqrt(2)
    am = abs(m)
    sg = -1.0 if am % 2 else 1.0  # (-1)^m Y^m = N P e^(i m phi)
    if m > 0:
        return [(am, sg * r2), (-am, r2)]
    return [(am, -1j * sg * r2), (-am, 1j * r2)]


def aabc(A, B, C, a, b, c, d, outer=48, method="sde"):
    """(ab|cd) for real normalised STFs a, b on A, c on B, d on C; functions are (n, l, m, zeta).

    Returns (value, imaginary part (should be ~0), number of radial quadrature nodes)."""
    R3 = [B[k] - A[k] for k in range(3)]
    R4 = [C[k] - A[k] for k in range(3)]
    eng = Engine(R3, R4, outer, method)
    total = 0j
    for ma, ca in real_to_complex(a[2]):
        for mb, cb in real_to_complex(b[2]):
            for mc, cc in real_to_complex(c[2]):
                for md, cd in real_to_complex(d[2]):
                    coef = complex(ca).conjugate() * cb * complex(cc).conjugate() * cd
                    total += coef * eng.sto_integral((a[0], a[1], ma, a[3]), (b[0], b[1], mb, b[3]),
                                                     (c[0], c[1], mc, c[3]), (d[0], d[1], md, d[3]))
    return total.real, total.imag, eng.nodes[0]


# ---------------------------------------------------------------------------------------------
# Examples with independent reference values
# ---------------------------------------------------------------------------------------------

ANG = 1 / 0.529177210903  # bohr per angstrom


def _water():
    r, th = 0.9572 * ANG, math.radians(104.52) / 2
    return [(0, 0, 0), (r * math.sin(th), 0, r * math.cos(th)), (-r * math.sin(th), 0, r * math.cos(th))]


def _ethylene():
    cc, ch, ang = 1.339 * ANG, 1.086 * ANG, math.radians(121.3)
    hz, hx = cc / 2 - ch * math.cos(ang), ch * math.sin(ang)
    return [(0, 0, cc / 2), (0, 0, -cc / 2), (hx, 0, -hz), (-hx, 0, -hz), (hx, 0, hz), (-hx, 0, hz)]


def _co2():
    co = 1.16 * ANG
    return [(0, 0, 0), (0, 0, co), (0, 0, -co)]


def _chloromethane():
    ccl, ch, hcl = 1.776 * ANG, 1.090 * ANG, math.radians(108.6)
    hz, hr = -ch * math.cos(math.pi - hcl), ch * math.sin(math.pi - hcl)
    return [(0, 0, 0), (0, 0, ccl)] + [(hr * math.cos(t), hr * math.sin(t), -hz)
                                       for t in (0, 2 * math.pi / 3, 4 * math.pi / 3)]


def _lioh():
    return [(0, 0, -1.582 * ANG), (0, 0, 0), (0, 0, 0.949 * ANG)]


# (label, geometry, indices of A, B, C, functions a, b, c, d as (n, l, m, zeta), reference value).
# Exponents are from the ADF DZP basis. Reference values: an independent code (exact one-centre
# potential of the AA density integrated on a fine molecular grid), converged to about 1e-15.
EXAMPLES = [
    ("H2O  (O 1s O 1s | H 1s H 1s), tight core on A", _water, (0, 1, 2),
     (1, 0, 0, 9.8), (1, 0, 0, 9.8), (1, 0, 0, 0.76), (1, 0, 0, 0.76), 2.7672300979425513e-01),
    ("H2O  (O 2px O 2px | H 1s H 1s)", _water, (0, 1, 2),
     (2, 1, 1, 1.3), (2, 1, 1, 3.06), (1, 0, 0, 0.76), (1, 0, 0, 1.28), 1.1933172356252013e-01),
    ("H2O  (H 1s H 1s | O 2px H 1s), p on B", _water, (1, 0, 2),
     (1, 0, 0, 0.76), (1, 0, 0, 0.76), (2, 1, 1, 1.3), (1, 0, 0, 0.76), -8.1763823603047500e-02),
    ("H2O  (O 3dz2 O 2pz | H 1s H 2px), d on A, p on C", _water, (0, 1, 2),
     (3, 2, 0, 2.0), (2, 1, 0, 3.06), (1, 0, 0, 0.76), (2, 1, 1, 1.25), 2.4153860413764196e-02),
    ("C2H4 (H 1s H 1s | C 2px C 2px), p on B and C", _ethylene, (4, 0, 1),
     (1, 0, 0, 0.76), (1, 0, 0, 0.76), (2, 1, 1, 0.96), (2, 1, 1, 0.96), 1.6865827648929535e-01),
    ("C2H4 (H 1s H 2pz | C 3dxz C 2pz), d on B, p on C", _ethylene, (4, 0, 1),
     (1, 0, 0, 0.76), (2, 1, 0, 1.25), (3, 2, 1, 2.2), (2, 1, 0, 2.2), 7.6285672783955706e-03),
    ("CO2  (O 2pz O 2pz | C 2pz O 1s), moment expansion", _co2, (1, 0, 2),
     (2, 1, 0, 1.3), (2, 1, 0, 1.3), (2, 1, 0, 2.2), (1, 0, 0, 9.8), -8.4793229530729583e-03),
    ("CH3Cl (H 1s H 1s | Cl 1s H 1s), tight core on B", _chloromethane, (2, 1, 3),
     (1, 0, 0, 0.76), (1, 0, 0, 0.76), (1, 0, 0, 20.3), (1, 0, 0, 0.76), 1.3544364600098090e-03),
    ("LiOH (Li 2pz Li 2pz | O 1s H 1s), diffuse on A, tight on B", _lioh, (0, 1, 2),
     (2, 1, 0, 0.61), (2, 1, 0, 0.61), (1, 0, 0, 9.8), (1, 0, 0, 0.76), 1.3570501416502779e-02),
]


def run_examples(outer, method):
    print(f"outer Gauss-Legendre points: {outer}, method: {method}")
    print(f"{'case':58s} {'this code':>24s} {'reference':>24s} {'difference':>11s} {'time':>7s}")
    for label, geom, (ia, ib, ic), a, b, c, d, ref in EXAMPLES:
        xyz = geom()
        t0 = time.time()
        val, _, _ = aabc(xyz[ia], xyz[ib], xyz[ic], a, b, c, d, outer, method)
        print(f"{label:58s} {val: .16e} {ref: .16e} {val - ref: .1e} {time.time() - t0:6.1f}s", flush=True)
        report_unstable()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Functions are 'n l m zeta' (real harmonics, m > 0 cos-type, "
                                        "m < 0 sin-type); coordinates in bohr.")
    for name, help_ in (("A", "centre of a and b"), ("B", "centre of c"), ("C", "centre of d")):
        ap.add_argument(f"--{name}", nargs=3, type=float, metavar=("X", "Y", "Z"), help=help_)
    for name in "abcd":
        ap.add_argument(f"--{name}", nargs=4, metavar=("N", "L", "M", "ZETA"), help=f"function {name}")
    ap.add_argument("--outer", type=int, default=48, help="Gauss-Legendre points over s (default 48, as in the paper)")
    ap.add_argument("--method", choices=("sde", "direct", "both"), default="sde",
                    help="radial quadrature: S + DE (default), direct DE, or both")
    ap.add_argument("--examples", action="store_true", help="run the bundled examples against reference values")
    args = ap.parse_args()

    methods = ("sde", "direct") if args.method == "both" else (args.method,)
    if args.examples:
        for m in methods:
            run_examples(args.outer, m)
        return
    if None in (args.A, args.B, args.C, args.a, args.b, args.c, args.d):
        ap.error("give --A --B --C and --a --b --c --d, or --examples")
    funcs = []
    for f in (args.a, args.b, args.c, args.d):
        n, l, m, z = int(f[0]), int(f[1]), int(f[2]), float(f[3])
        if not (n > l >= 0 and abs(m) <= l and z > 0):
            ap.error(f"invalid function {' '.join(f)}")
        funcs.append((n, l, m, z))
    for m in methods:
        t0 = time.time()
        val, imag, nodes = aabc(args.A, args.B, args.C, *funcs, outer=args.outer, method=m)
        print(f"{m:6s} (ab|cd) = {val: .16e}   imag {imag:.1e}   radial nodes {nodes}   {time.time() - t0:.2f} s")
        report_unstable()


if __name__ == "__main__":
    main()
