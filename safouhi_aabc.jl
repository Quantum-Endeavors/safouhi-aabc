#!/usr/bin/env julia
#=
Three-centre (AA|BC) two-electron Coulomb integrals over Slater-type functions by the method of

    H. Safouhi, "The S and Double-Exponential Transformations with a Moment Expansion for
    Three-Centre Coulomb and Hybrid Integrals", arXiv:2611.23965v1.

    (ab|cd) = ∫∫ a(r1) b(r1) (1 / |r1 - r2|) c(r2) d(r2) dr1 dr2

with a and b on centre A, c on centre B, d on centre C (three distinct centres).

Steps:
  - each normalised STF is a finite sum of B functions (paper eq. 11);
  - each B-function integral uses the Fourier-B reduction (eqs. 17, 18): finite angular sums,
    an outer Feynman integral over s in [0, 1] (Gauss-Legendre) and a semi-infinite radial
    integral K(s) containing j_λ(v x) (eq. 19);
  - K(s) is evaluated with the S transformation (eq. 26) and the Ooura-Mori Fourier-type DE
    quadrature (φ₁, K = 6, M h = π / v, eqs. 35-36), or with the moment expansion (eq. 68)
    when ρ(s) <= 0.5 (<= 0.9 for λ >= 3), section 7. "Direct DE" on j_λ (eq. 40) is
    available as a cross-check (--method direct / both).

Departures from the printed formulas, needed to reproduce independent reference values:
  1. Eq. 18: Δl = (l3' + l4' - l') / 2. The printed (l3 + l4 - l') / 2 is not an integer
     when l3' < l3.
  2. Eq. 17: the factor (-1)^((l3'+l4'+l')/2) Y_l'^(m3'-m4')(R34-hat) is replaced by
     Y_l'^(m4'-m3')(R34-hat), i.e. the Gaunt coefficient's own M and no extra sign, with
     R34 = R3 - R4. As printed, the result is wrong whenever B or C carries a p_z or d
     function, or both carry p functions.
  3. The number of Gauss-Legendre points for the s integral is an option (--outer). The
     paper's 48 points are not enough when the B and C exponents are very different (e.g. a
     tight core 1s on B); the integrand then peaks sharply near s = 0. --outer 384 is safe
     for all bundled examples.
  4. M (and hence h = π / (v M)) is chosen per radial integral by doubling M from 2 until two
     successive values agree to 1e-14 relative, instead of being fixed per case.

Functions are normalised real STFs N r^(n-1) e^(-ζ r) Z_lm, where Z_lm are real spherical
harmonics without the Condon-Shortley phase: m > 0 ~ cos(m φ) (p: m=1 is x), m < 0 ~
sin(|m| φ) (p: m=-1 is y), m = 0 (p: z). Coordinates are in bohr, results in hartree.

Only the Julia standard library is used.

Usage:
  julia safouhi_aabc.jl --A 0 0 0 --B 1.43 0 1.11 --C -1.43 0 1.11 \
      --a 1 0 0 9.8 --b 1 0 0 9.8 --c 1 0 0 0.76 --d 1 0 0 0.76
  julia safouhi_aabc.jl --examples [--outer 384] [--method sde|direct|both]

Each function is given as "n l m zeta". From the REPL: include("safouhi_aabc.jl") and call
aabc(A, B, C, a, b, c, d; outer = 48, method = :sde) with a = (n, l, m, ζ) etc.
=#

using Printf

# ---------------------------------------------------------------------------------------------
# Small special functions
# ---------------------------------------------------------------------------------------------

const FACT = [Float64(factorial(big(n))) for n in 0:170]
fact(n::Integer) = n < 0 ? throw(DomainError(n, "factorial of a negative number")) : FACT[n+1]
sgn(k::Integer) = iseven(k) ? 1.0 : -1.0  # (-1)^k

"n!! for n >= -1."
function dfact(n::Integer)
    r = 1.0
    while n > 1
        r *= n
        n -= 2
    end
    r
end

"Generalised binomial coefficient (a choose k)."
binom_real(a, k) = prod(((a - j) / (j + 1) for j in 0:k-1); init = 1.0)

pochhammer(a, n) = prod((a + j for j in 0:n-1); init = 1.0)

"Wigner 3j symbol (Racah formula), integer arguments."
function wigner3j(j1, j2, j3, m1, m2, m3)
    (m1 + m2 + m3 != 0 || j3 < abs(j1 - j2) || j3 > j1 + j2) && return 0.0
    (abs(m1) > j1 || abs(m2) > j2 || abs(m3) > j3) && return 0.0
    tri = fact(j1 + j2 - j3) * fact(j1 - j2 + j3) * fact(-j1 + j2 + j3) / fact(j1 + j2 + j3 + 1)
    pre = sqrt(tri * fact(j1 + m1) * fact(j1 - m1) * fact(j2 + m2) * fact(j2 - m2) *
               fact(j3 + m3) * fact(j3 - m3))
    total = 0.0
    for k in max(0, j2 - j3 - m1, j1 - j3 + m2):min(j1 + j2 - j3, j1 - m1, j2 + m2)
        den = fact(k) * fact(j1 + j2 - j3 - k) * fact(j1 - m1 - k) * fact(j2 + m2 - k) *
              fact(j3 - j2 + m1 + k) * fact(j3 - j1 - m2 + k)
        total += sgn(k) / den
    end
    sgn(j1 - j2 - m3) * pre * total
end

const GAUNT_CACHE = Dict{NTuple{6,Int},Float64}()

"Paper eq. 9: ∫ conj(Y_l1^m1) Y_l2^m2 Y_L^M dΩ (Condon-Shortley Y)."
gaunt(l1, m1, l2, m2, L, M) = get!(GAUNT_CACHE, (l1, m1, l2, m2, L, M)) do
    (m1 != m2 + M || L < abs(l1 - l2) || L > l1 + l2 || isodd(l1 + l2 + L)) && return 0.0
    (abs(m1) > l1 || abs(m2) > l2 || abs(M) > L) && return 0.0
    sgn(m1) * sqrt((2l1 + 1) * (2l2 + 1) * (2L + 1) / 4π) *
        wigner3j(l1, l2, L, 0, 0, 0) * wigner3j(l1, l2, L, -m1, m2, M)
end

"Associated Legendre P_l^m (m >= 0) without the Condon-Shortley phase."
function plm(l, m, x)
    pmm = 1.0
    s = sqrt(max(0.0, (1 - x) * (1 + x)))
    for i in 1:m
        pmm *= (2i - 1) * s
    end
    l == m && return pmm
    pmmp1 = x * (2m + 1) * pmm
    l == m + 1 && return pmmp1
    pll = 0.0
    for ll in m+2:l
        pll = ((2ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm, pmmp1 = pmmp1, pll
    end
    pll
end

"Paper eq. 2: Y_l^m = i^(m+|m|) N P_l^|m|(cos θ) e^(i m φ)."
function ylm(l, m, ct, φ)
    am = abs(m)
    n = sqrt((2l + 1) / 4π * fact(l - am) / fact(l + am))
    ph = (m > 0 && isodd(m)) ? -1.0 : 1.0
    ph * n * plm(l, am, ct) * cis(m * φ)
end

"Regular solid harmonic r^l Y_l^m(r-hat), finite at r = 0."
function solid(l, m, v)
    r = sqrt(v[1]^2 + v[2]^2 + v[3]^2)
    r == 0 && return l == 0 ? complex(1 / sqrt(4π)) : 0.0im
    r^l * ylm(l, m, v[3] / r, atan(v[2], v[1]))
end

"Spherical Bessel j_l(z), z >= 0."
function sph_j(l, z)
    if z < 0.5 + l  # power series, eq. 7
        term = z^l / dfact(2l + 1)
        total = term
        for q in 1:199
            term *= -z * z / (2q * (2l + 2q + 1))
            total += term
            abs(term) < 1e-18 * abs(total) && break
        end
        return total
    end
    j0 = sin(z) / z
    l == 0 && return j0
    j1 = sin(z) / z^2 - cos(z) / z
    for k in 1:l-1
        j0, j1 = j1, (2k + 1) / z * j1 - j0
    end
    j1
end

"Reduced Bessel k-hat_{N+1/2}(z) = e^-z Σ_p c_p z^p; returns c_0 .. c_N."
khat_poly(N) = [fact(2N - p) / (fact(N - p) * fact(p)) / 2.0^(N - p) for p in 0:N]

# ---------------------------------------------------------------------------------------------
# Truncated Taylor series ("jets") in u = x², for (1/x d/dx)^λ = 2^λ d^λ/du^λ.
# A jet c holds c[k+1] = coefficient of δ^k.
# ---------------------------------------------------------------------------------------------

function jet_mul(a, b)
    n = length(a)
    c = zeros(n)
    for i in 1:n, j in 1:n-i+1
        c[i+j-1] += a[i] * b[j]
    end
    c
end

"(c0 + c1 δ)^p to order n, c0 > 0."
jet_pow_linear(c0, c1, p, n) = [binom_real(p, k) * c0^(p - k) * c1^k for k in 0:n]

"exp(-z) for a jet z."
function jet_exp_neg(z)
    n = length(z) - 1
    e = zeros(n + 1)
    e[1] = 1.0
    for k in 1:n
        e[k+1] = sum(j * (-z[j+1]) * e[k-j+1] for j in 1:k) / k
    end
    e .* exp(-z[1])
end

# ---------------------------------------------------------------------------------------------
# The semi-infinite radial integral K(s), eq. 19, divided by v^λ (the v^λ goes with
# Y_λ^μ(v-hat) into a solid harmonic, so v = 0 needs no special case).
# ---------------------------------------------------------------------------------------------

struct Radial
    nx::Int
    nk::Int
    ngamma::Int
    nu_int::Int  # ν = nu_int + 1/2
    lam::Int
    zs::Float64
    z3::Float64
    z4::Float64
    R34::Float64
end

"x^nx k-hat_ν(R34 γ) / ((zs² + x²)^nk γ^ngamma), without the oscillatory factor."
function amplitude(p::Radial, s, x, kp)
    g = sqrt((1 - s) * p.z4^2 + s * p.z3^2 + s * (1 - s) * x * x)
    z = p.R34 * g
    poly = evalpoly(z, kp)
    # in logs: at the far exp-sinh nodes x^nx overflows before e^-z underflows
    exp(p.nx * log(x) - z + log(poly) - p.nk * log(p.zs^2 + x * x) - p.ngamma * log(g))
end

"G_{λ,s}(x) = (1/x d/dx)^λ [x^(nx+λ-1) k-hat / ((zs²+x²)^nk γ^ngamma)], eq. 26."
function g_lambda(p::Radial, s, x, kp)
    L = p.lam
    u = x * x
    f = jet_pow_linear(u, 1.0, 0.5 * (p.nx + L - 1), L)                # u^a
    f = jet_mul(f, jet_pow_linear(p.zs^2 + u, 1.0, -p.nk, L))          # (zs² + u)^-nk
    G0 = (1 - s) * p.z4^2 + s * p.z3^2 + s * (1 - s) * u
    g1 = s * (1 - s)
    f = jet_mul(f, jet_pow_linear(G0, g1, -0.5 * p.ngamma, L))         # γ^-ngamma
    z = p.R34 .* jet_pow_linear(G0, g1, 0.5, L)                        # R34 γ
    poly = zeros(L + 1)
    zp = [1.0; zeros(L)]
    for c in kp                                                        # Σ c_i z^i
        poly .+= c .* zp
        zp = jet_mul(zp, z)
    end
    f = jet_mul(f, jet_mul(poly, jet_exp_neg(z)))
    2.0^L * fact(L) * f[L+1]
end

"Ooura-Mori φ₁(t) = t / (1 - exp(-K sinh t)) and its derivative."
function phi1(t, K = 6.0)
    abs(t) < 1e-8 && return 1 / K + 0.5t, 0.5
    om = -expm1(-K * sinh(t))  # 1 - E
    (!isfinite(om) || abs(om) > 1e300) && return 0.0, 0.0
    E = 1 - om
    t / om, (om - t * K * cosh(t) * E) / om^2
end

"Fourier-type DE sum M h Σ_k f(M φ(kh)) φ'(kh), eq. 32."
function de_fourier(f, M, h, counter)
    total = 0.0
    for dir in (1, -1)
        small = 0
        k = dir > 0 ? 0 : -1
        while abs(k) < 200000
            ph, dph = phi1(k * h)
            # x -> 0 double exponentially; stop before x² underflows (the weight φ' is ~0 there)
            (ph <= 0 || dph == 0 || M * ph < 1e-100) && break
            term = f(M * ph) * dph
            counter[] += 1
            total += term
            scale = total != 0 ? abs(total) : 1.0
            small = abs(term) * M * h < 1e-17 * scale ? small + 1 : 0
            small >= 4 && break
            k += dir
        end
    end
    M * h * total
end

"exp-sinh quadrature on [0, ∞) for a non-oscillatory integrand, refined until stable."
function exp_sinh(f, counter)
    prev = 0.0
    h = 0.5
    while h > 1e-3
        total = 0.0
        for dir in (1, -1)
            small = 0
            k = dir > 0 ? 0 : -1
            while abs(k) < 100000
                t = k * h
                ex = 0.5π * sinh(t)
                abs(ex) > 700 && break
                x = exp(ex)
                term = f(x) * 0.5π * cosh(t) * x
                counter[] += 1
                total += term
                small = abs(term) < 1e-18 * abs(total) ? small + 1 : 0
                small >= 3 && break
                k += dir
            end
        end
        total *= h
        h < 0.5 && abs(total - prev) <= 1e-15 * abs(total) && return total
        prev = total
        h *= 0.5
    end
    prev
end

# radial integrals that never stabilised to 1e-14 (count, worst relative change at M = 256)
const UNSTABLE = Ref((0, 0.0))

function report_unstable()
    n, worst = UNSTABLE[]
    n > 0 && @printf("    note: %d radial integrals changed by up to %.1e (relative) between M = 128 and 256\n",
                     n, worst)
    UNSTABLE[] = (0, 0.0)
end

"K(s) / v^λ."
function radial(p::Radial, s, v, method, counter)
    kp = khat_poly(p.nu_int)
    L = p.lam
    ρ = v / (p.R34 * sqrt(s * (1 - s)))
    if ρ <= (L >= 3 ? 0.9 : 0.5)  # moment expansion, eq. 68
        total = 0.0
        for q in 0:399
            c = sgn(q) * v^(2q) / (2.0^q * fact(q) * dfact(2L + 2q + 1))
            pq = Radial(p.nx + L + 2q, p.nk, p.ngamma, p.nu_int, p.lam, p.zs, p.z3, p.z4, p.R34)
            term = c * exp_sinh(x -> amplitude(pq, s, x, kp), counter)
            total += term
            abs(term) < 1e-17 * abs(total) && q > 2 && return total
        end
        @printf(stderr, "warning: moment series not converged (rho %.3f)\n", ρ)
        return total
    end
    # choose M by doubling until two successive values agree
    prev = prev_prev = val = 0.0
    M = 2.0
    while M <= 256
        if method == :sde
            h = π / (v * M)  # eq. 35
            val = de_fourier(x -> g_lambda(p, s, x, kp) * sin(v * x), M, h, counter)
            val /= v^(2L + 1)  # eq. 26: v^-(λ+1), and / v^λ
        else
            h = π / (2v * M)  # eq. 40
            val = de_fourier(x -> amplitude(p, s, x, kp) * sph_j(L, v * x), M, h, counter)
            val /= v^L
        end
        M > 2 && abs(val - prev) <= 1e-14 * abs(val) + 1e-300 && return val
        prev_prev, prev = prev, val
        M *= 2
    end
    n, worst = UNSTABLE[]
    UNSTABLE[] = (n + 1, max(worst, val != 0 ? abs(val - prev_prev) / abs(val) : 0.0))
    prev
end

"n-point Gauss-Legendre nodes and weights on [0, 1]."
function gauss_legendre01(n)
    xs, ws = zeros(n), zeros(n)
    for i in 0:n-1
        z = cos(π * (i + 0.75) / (n + 0.5))
        dp = 0.0
        for _ in 1:100
            p0, p1 = 1.0, z
            for k in 2:n
                p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
            end
            dp = n * (z * p1 - p0) / (z * z - 1)
            dz = p1 / dp
            z -= dz
            abs(dz) < 1e-16 && break
        end
        xs[i+1] = 0.5 * (1 - z)
        ws[i+1] = 1 / ((1 - z * z) * dp^2)
    end
    xs, ws
end

# ---------------------------------------------------------------------------------------------
# B-function integral, eq. 17
# ---------------------------------------------------------------------------------------------

const BFun = Tuple{Int,Int,Int,Float64}  # (n, l, m, ζ)

ipow(k) = (1.0 + 0im, 1.0im, -1.0 + 0im, -1.0im)[mod(k, 4)+1]

struct Engine
    R3::Vector{Float64}  # centres B and C relative to A
    R4::Vector{Float64}
    method::Symbol
    sx::Vector{Float64}
    sw::Vector{Float64}
    v::Vector{Vector{Float64}}  # v(s) = (1 - s) R3 + s R4 at the s nodes
    cache::Dict{Radial,Vector{Float64}}
    nodes::Base.RefValue{Int}
end

function Engine(R3, R4; outer = 48, method = :sde)
    sx, sw = gauss_legendre01(outer)
    Engine(R3, R4, method, sx, sw, [(1 - s) .* R3 .+ s .* R4 for s in sx], Dict{Radial,Vector{Float64}}(), Ref(0))
end

"K(s) / v^λ at the s nodes, cached on the radial parameters."
radial_table(e::Engine, p::Radial) = get!(e.cache, p) do
    [radial(p, s, sqrt(sum(abs2, v)), e.method, e.nodes) for (s, v) in zip(e.sx, e.v)]
end

"Eq. 17 for B functions b = (n, l, m, ζ): [B1]* B2 on A, [B3]* on B, B4 on C."
function bfun_integral(e::Engine, b1::BFun, b2::BFun, b3::BFun, b4::BFun)
    n1, l1, m1, z1 = b1
    n2, l2, m2, z2 = b2
    n3, l3, m3, z3 = b3
    n4, l4, m4, z4 = b4
    zs = z1 + z2
    R34v = e.R3 .- e.R4
    R34 = sqrt(sum(abs2, R34v))
    pre = 8 * (4π)^3 * sqrt(π) * z1^l1 * z2^l2 * z3^(2n3 + l3 - 1) * z4^(2n4 + l4 - 1) *
          dfact(2l3 + 1) * dfact(2l4 + 1) * fact(n3 + l3 + n4 + l4 + 1) /
          (2.0^(l1 + l2 + 1) * fact(n1 + l1) * fact(n2 + l2) * fact(n3 + l3) * fact(n4 + l4))
    total = 0.0im
    for l in abs(l1 - l2):l1+l2
        gl = gaunt(l2, m2, l1, m1, l, m2 - m1)
        gl == 0 && continue
        tl = (-1.0im)^l * gl / 2.0^(2n1 + 2n2 + l)
        for k in 2:n1+n2
            ck = sum((2.0^k * fact(2n1 - i - 1) * fact(2n2 - k + i - 1) * z1^(i - 1) * z2^(k - i - 1) /
                      (fact(i - 1) * fact(n1 - i) * fact(k - i - 1) * fact(n2 - k + i))
                      for i in max(1, k - n2):min(n1, k - 1)); init = 0.0)
            ck == 0 && continue
            nk = k + l1 + l2
            η = l - k - l1 - l2 + 1
            ηp = iseven(η) ? -η ÷ 2 : -(η + 1) ÷ 2
            fk = zs^(nk - l - 1) * fact(k + l1 + l2 + l) / gamma_half(l)
            for l4p in 0:l4, m4p in -l4p:l4p
                abs(m4 - m4p) > l4 - l4p && continue
                g4 = gaunt(l4, m4, l4 - l4p, m4 - m4p, l4p, m4p)
                g4 == 0 && continue
                f4 = ipow(l4 + l4p) * sgn(l4p) * g4 / (dfact(2l4p + 1) * dfact(2(l4 - l4p) + 1))
                for l3p in 0:l3, m3p in -l3p:l3p
                    abs(m3 - m3p) > l3 - l3p && continue
                    g3 = gaunt(l3, m3, l3 - l3p, m3 - m3p, l3p, m3p)
                    g3 == 0 && continue
                    f3 = ipow(l3 + l3p) * g3 / (dfact(2l3p + 1) * dfact(2(l3 - l3p) + 1))
                    for lp in 0:l3p+l4p
                        gp = gaunt(l4p, m4p, l3p, m3p, lp, m4p - m3p)
                        gp == 0 && continue
                        # correction 2: printed (-1)^((l3'+l4'+l')/2) Y_l'^(m3'-m4')(R34-hat)
                        fp = gp * solid(lp, m4p - m3p, R34v)
                        m34 = (m3 - m3p) - (m4 - m4p)
                        μ = (m2 - m1) - m34
                        # correction 1: Δl = (l3' + l4' - l') / 2
                        Δl = (l3p + l4p - lp) ÷ 2
                        for l34 in 0:(l3-l3p)+(l4-l4p)
                            g34 = gaunt(l3 - l3p, m3 - m3p, l4 - l4p, m4 - m4p, l34, m34)
                            g34 == 0 && continue
                            for λ in 0:l+l34
                                gla = gaunt(l, m2 - m1, l34, m34, λ, μ)
                                gla == 0 && continue
                                flam = ipow(λ) * gla
                                ysol = [solid(λ, μ, v) for v in e.v]
                                for j in 0:Δl
                                    fj = binom_real(Δl, j) * sgn(j) /
                                         (2.0^(n3 + n4 + l3 + l4 - j + 1) * fact(n3 + n4 + l3 + l4 - j + 1))
                                    nu_int = n3 + n4 + l3 + l4 - lp - j
                                    ngamma = 2(n3 + l3 + n4 + l4) - (l3p + l4p) - lp + 1
                                    for r in 0:ηp
                                        fr = sgn(r) * pochhammer(η / 2, r) * pochhammer((η + 1) / 2, r) /
                                             (pochhammer(l + 1.5, r) * fact(r) * zs^(2r))
                                        fr == 0 && continue
                                        p = Radial(l3 - l3p + l4 - l4p + 2r + l, nk, ngamma, nu_int, λ, zs, z3, z4, R34)
                                        kv = radial_table(e, p)
                                        sint = sum(e.sw[i] * e.sx[i]^(n3 + l3 + l4 - l4p) * (1 - e.sx[i])^(n4 + l4 + l3 - l3p) *
                                                   ysol[i] * kv[i] for i in eachindex(e.sx))
                                        total += tl * ck * fk * f4 * f3 * fp * g34 * flam * fj * fr * sint
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    pre * total
end

"Γ(l + 3/2) = (2l+1)!! √π / 2^(l+1)."
gamma_half(l) = dfact(2l + 1) * sqrt(π) / 2.0^(l + 1)

"Normalised STF (n, l, complex m, ζ) as a sum of B functions, eq. 11."
function sto_to_b(n, l, m, ζ)
    norm = sqrt((2ζ)^(2n + 1) / fact(2n)) / ζ^(n - 1)
    [(norm * sgn(n - l - p) * fact(n - l) * 2.0^(2p + 2l - n) * fact(l + p) / (fact(2p - n + l) * fact(n - l - p)),
      (p, l, m, Float64(ζ))) for p in (n-l+1)÷2:n-l]
end

"Complex-harmonic STF integral [χ1]* χ2 [χ3]* χ4."
function sto_integral(e::Engine, s1, s2, s3, s4)
    total = 0.0im
    for (c1, b1) in sto_to_b(s1...), (c2, b2) in sto_to_b(s2...), (c3, b3) in sto_to_b(s3...), (c4, b4) in sto_to_b(s4...)
        total += c1 * c2 * c3 * c4 * bfun_integral(e, b1, b2, b3, b4)
    end
    total
end

"Real harmonic (m > 0 ~ √2 cos, m < 0 ~ √2 sin, no Condon-Shortley phase) as (m, coefficient) pairs of Y_l^m."
function real_to_complex(m)
    m == 0 && return [(0, 1.0 + 0im)]
    r2 = 1 / sqrt(2)
    am = abs(m)
    sg = isodd(am) ? -1.0 : 1.0  # (-1)^m Y^m = N P e^(i m φ)
    m > 0 ? [(am, complex(sg * r2)), (-am, complex(r2))] : [(am, -1im * sg * r2), (-am, 1im * r2)]
end

"""
    aabc(A, B, C, a, b, c, d; outer = 48, method = :sde)

(ab|cd) for real normalised STFs a, b on A, c on B, d on C; functions are (n, l, m, ζ).
Returns (value, imaginary part (should be ~0), number of radial quadrature nodes).
"""
function aabc(A, B, C, a, b, c, d; outer = 48, method = :sde)
    e = Engine(collect(Float64, B .- A), collect(Float64, C .- A); outer, method)
    total = 0.0im
    for (ma, ca) in real_to_complex(a[3]), (mb, cb) in real_to_complex(b[3]),
        (mc, cc) in real_to_complex(c[3]), (md, cd) in real_to_complex(d[3])
        total += conj(ca) * cb * conj(cc) * cd *
                 sto_integral(e, (a[1], a[2], ma, a[4]), (b[1], b[2], mb, b[4]), (c[1], c[2], mc, c[4]), (d[1], d[2], md, d[4]))
    end
    real(total), imag(total), e.nodes[]
end

# ---------------------------------------------------------------------------------------------
# Examples with independent reference values
# ---------------------------------------------------------------------------------------------

const ANG = 1 / 0.529177210903  # bohr per ångström

function water()
    r, θ = 0.9572ANG, deg2rad(104.52) / 2
    [[0.0, 0, 0], [r * sin(θ), 0, r * cos(θ)], [-r * sin(θ), 0, r * cos(θ)]]
end

function ethylene()
    cc, ch, ang = 1.339ANG, 1.086ANG, deg2rad(121.3)
    hz, hx = cc / 2 - ch * cos(ang), ch * sin(ang)
    [[0.0, 0, cc / 2], [0.0, 0, -cc / 2], [hx, 0, -hz], [-hx, 0, -hz], [hx, 0, hz], [-hx, 0, hz]]
end

co2() = [[0.0, 0, 0], [0.0, 0, 1.16ANG], [0.0, 0, -1.16ANG]]

function chloromethane()
    ccl, ch, hcl = 1.776ANG, 1.090ANG, deg2rad(108.6)
    hz, hr = -ch * cos(π - hcl), ch * sin(π - hcl)
    [[[0.0, 0, 0], [0.0, 0, ccl]]; [[hr * cos(t), hr * sin(t), -hz] for t in (0, 2π / 3, 4π / 3)]]
end

lioh() = [[0.0, 0, -1.582ANG], [0.0, 0, 0], [0.0, 0, 0.949ANG]]

# (label, geometry, indices of A, B, C (1-based), functions a, b, c, d as (n, l, m, ζ), reference value).
# Exponents are from the ADF DZP basis. Reference values: an independent code (exact one-centre
# potential of the AA density integrated on a fine molecular grid), converged to about 1e-15.
const EXAMPLES = [
    ("H2O  (O 1s O 1s | H 1s H 1s), tight core on A", water, (1, 2, 3),
     (1, 0, 0, 9.8), (1, 0, 0, 9.8), (1, 0, 0, 0.76), (1, 0, 0, 0.76), 2.7672300979425513e-01),
    ("H2O  (O 2px O 2px | H 1s H 1s)", water, (1, 2, 3),
     (2, 1, 1, 1.3), (2, 1, 1, 3.06), (1, 0, 0, 0.76), (1, 0, 0, 1.28), 1.1933172356252013e-01),
    ("H2O  (H 1s H 1s | O 2px H 1s), p on B", water, (2, 1, 3),
     (1, 0, 0, 0.76), (1, 0, 0, 0.76), (2, 1, 1, 1.3), (1, 0, 0, 0.76), -8.1763823603047500e-02),
    ("H2O  (O 3dz2 O 2pz | H 1s H 2px), d on A, p on C", water, (1, 2, 3),
     (3, 2, 0, 2.0), (2, 1, 0, 3.06), (1, 0, 0, 0.76), (2, 1, 1, 1.25), 2.4153860413764196e-02),
    ("C2H4 (H 1s H 1s | C 2px C 2px), p on B and C", ethylene, (5, 1, 2),
     (1, 0, 0, 0.76), (1, 0, 0, 0.76), (2, 1, 1, 0.96), (2, 1, 1, 0.96), 1.6865827648929535e-01),
    ("C2H4 (H 1s H 2pz | C 3dxz C 2pz), d on B, p on C", ethylene, (5, 1, 2),
     (1, 0, 0, 0.76), (2, 1, 0, 1.25), (3, 2, 1, 2.2), (2, 1, 0, 2.2), 7.6285672783955706e-03),
    ("CO2  (O 2pz O 2pz | C 2pz O 1s), moment expansion", co2, (2, 1, 3),
     (2, 1, 0, 1.3), (2, 1, 0, 1.3), (2, 1, 0, 2.2), (1, 0, 0, 9.8), -8.4793229530729583e-03),
    ("CH3Cl (H 1s H 1s | Cl 1s H 1s), tight core on B", chloromethane, (3, 2, 4),
     (1, 0, 0, 0.76), (1, 0, 0, 0.76), (1, 0, 0, 20.3), (1, 0, 0, 0.76), 1.3544364600098090e-03),
    ("LiOH (Li 2pz Li 2pz | O 1s H 1s), diffuse on A, tight on B", lioh, (1, 2, 3),
     (2, 1, 0, 0.61), (2, 1, 0, 0.61), (1, 0, 0, 9.8), (1, 0, 0, 0.76), 1.3570501416502779e-02),
]

function run_examples(outer, method)
    println("outer Gauss-Legendre points: $outer, method: $method")
    @printf("%-58s %24s %24s %11s %7s\n", "case", "this code", "reference", "difference", "time")
    for (label, geom, (ia, ib, ic), a, b, c, d, ref) in EXAMPLES
        xyz = geom()
        t = @elapsed val, _, _ = aabc(xyz[ia], xyz[ib], xyz[ic], a, b, c, d; outer, method)
        @printf("%-58s %24.16e %24.16e %11.1e %6.1fs\n", label, val, ref, val - ref, t)
        flush(stdout)
        report_unstable()
    end
end

const USAGE = """
usage: julia safouhi_aabc.jl --A X Y Z --B X Y Z --C X Y Z --a N L M ZETA --b ... --c ... --d ...
                             [--outer 48] [--method sde|direct|both]
       julia safouhi_aabc.jl --examples [--outer 48] [--method sde|direct|both]

(ab|cd) with a, b on A, c on B, d on C. Functions are 'n l m zeta' (real harmonics, m > 0 cos-type,
m < 0 sin-type); coordinates in bohr. --outer: Gauss-Legendre points over s (default 48, as in the
paper). --method: radial quadrature, S + DE (default), direct DE, or both.
"""

function main(args)
    opts = Dict{String,Vector{String}}()
    nargs = Dict("--A" => 3, "--B" => 3, "--C" => 3, "--a" => 4, "--b" => 4, "--c" => 4, "--d" => 4,
                 "--outer" => 1, "--method" => 1, "--examples" => 0, "--help" => 0, "-h" => 0)
    i = 1
    while i <= length(args)
        key = args[i]
        haskey(nargs, key) || (print(stderr, "unknown option $key\n\n", USAGE); exit(2))
        n = nargs[key]
        i + n <= length(args) || (print(stderr, "$key needs $n values\n\n", USAGE); exit(2))
        opts[key] = args[i+1:i+n]
        i += n + 1
    end
    if haskey(opts, "--help") || haskey(opts, "-h")
        print(USAGE)
        return
    end
    outer = parse(Int, get(opts, "--outer", ["48"])[1])
    m = get(opts, "--method", ["sde"])[1]
    m in ("sde", "direct", "both") || (print(stderr, "bad --method $m\n\n", USAGE); exit(2))
    methods = m == "both" ? (:sde, :direct) : (Symbol(m),)
    if haskey(opts, "--examples")
        foreach(meth -> run_examples(outer, meth), methods)
        return
    end
    all(k -> haskey(opts, k), ("--A", "--B", "--C", "--a", "--b", "--c", "--d")) ||
        (print(stderr, "give --A --B --C and --a --b --c --d, or --examples\n\n", USAGE); exit(2))
    A, B, C = (parse.(Float64, opts[k]) for k in ("--A", "--B", "--C"))
    funcs = map(("--a", "--b", "--c", "--d")) do k
        f = opts[k]
        n, l, mm, z = parse(Int, f[1]), parse(Int, f[2]), parse(Int, f[3]), parse(Float64, f[4])
        (n > l >= 0 && abs(mm) <= l && z > 0) || (print(stderr, "invalid function $(join(f, ' '))\n"); exit(2))
        (n, l, mm, z)
    end
    for meth in methods
        t = @elapsed val, im_, nodes = aabc(A, B, C, funcs...; outer, method = meth)
        @printf("%-6s (ab|cd) = %23.16e   imag %.1e   radial nodes %d   %.2f s\n", meth, val, im_, nodes, t)
        report_unstable()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
