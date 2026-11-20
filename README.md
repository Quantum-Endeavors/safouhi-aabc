# (AA|BC) Slater-type integrals by the S + DE method

A single-file implementation of the method in

> H. Safouhi, *The S and Double-Exponential Transformations with a Moment Expansion for
> Three-Centre Coulomb and Hybrid Integrals*, [arXiv:2611.23965v1](https://arxiv.org/abs/2611.23965v1)

for one three-centre two-electron Coulomb integral over normalised Slater-type functions:

    (ab|cd) = ∫∫ a(r₁) b(r₁) |r₁ − r₂|⁻¹ c(r₂) d(r₂) dr₁ dr₂,    a, b on A;  c on B;  d on C.

`safouhi_aabc.jl` (Julia) is the main program. `safouhi_aabc.py` (Python) is a line-by-line
equivalent that serves as a cross-check. Both use only their standard libraries.

## Running

Julia ≥ 1.6:

```sh
julia safouhi_aabc.jl --A 0 0 0 --B 1.43 0 1.11 --C -1.43 0 1.11 \
    --a 2 1 1 1.3 --b 3 2 1 2.0 --c 2 1 0 1.25 --d 1 0 0 0.76 --method both
julia safouhi_aabc.jl --examples --outer 384
```

From the REPL:

```julia
include("safouhi_aabc.jl")
aabc([0, 0, 0], [1.43, 0, 1.11], [-1.43, 0, 1.11],
     (1, 0, 0, 9.8), (1, 0, 0, 9.8), (1, 0, 0, 0.76), (1, 0, 0, 0.76); outer = 384)
# -> (value, imaginary part, number of radial quadrature nodes)
```

Python ≥ 3.8 takes the same options:

```sh
python3 safouhi_aabc.py --examples --outer 384
```

- Each function is given as `n l m zeta`.
- Real spherical harmonics are used, without the Condon–Shortley phase:
  - m > 0 is cos-type (for p, m = 1 is p_x);
  - m < 0 is sin-type (m = −1 is p_y);
  - m = 0 is p_z.
- Coordinates are in bohr and results in hartree.
- `--outer` sets the number of Gauss–Legendre points for the Feynman integral over s. The default is the paper's 48.
- `--method` chooses the radial quadrature: `sde` (default), `direct` (direct DE on j_λ, eq. 40), or `both`.

## What is implemented

- **STF to B functions** (eq. 11).
- **Fourier–B reduction** (eqs. 17–18). This gives:
  - finite angular sums with Gaunt coefficients;
  - an outer integral over s;
  - the semi-infinite radial integral K(s) (eq. 19).
- **K(s) by S + DE**:
  - The S transformation (eq. 26) moves (1/x d/dx)^λ onto the non-oscillatory part. Its derivatives are computed exactly with truncated Taylor series in u = x².
  - The Ooura–Mori φ₁ Fourier-type DE rule then follows (K = 6, M h = π / v; eqs. 35–36).
- **Moment expansion** (eq. 68) when ρ(s) ≤ 0.5, or ρ(s) ≤ 0.9 for λ ≥ 3 (section 7). Its moments are computed with an exp–sinh rule.
- **Direct DE** on j_λ (eq. 40), as an independent check of the S transformation.

## Departures from the printed formulas

1. **Eq. 18:** Δl is taken as (l₃′ + l₄′ − l′)/2. The printed (l₃ + l₄ − l′)/2 is not an integer when l₃′ < l₃, for example for a p function on B with l₃′ = 0.
2. **Eq. 17:** the factor (−1)^((l₃′+l₄′+l′)/2) Y_{l′}^{m₃′−m₄′}(R̂₃₄) is replaced by Y_{l′}^{m₄′−m₃′}(R̂₃₄), with R₃₄ = R₃ − R₄.
   - That is, the replacement uses the M of its own Gaunt coefficient ⟨l₄′m₄′|l₃′m₃′|l′ m₄′−m₃′⟩ and no extra sign.
   - As printed, the results are wrong whenever B or C carries a p_z or d function, or both carry p functions (errors up to 0.26 hartree).
   - With the change, 25 p–p and d–p test integrals over all three centres agree with the reference to ≤ 1.4 × 10⁻¹⁵.
   - These two changes were found from symmetry and from comparison with the reference code, not derived. A check against the derivation would be welcome.
3. **The outer integral over s:**
   - 48 Gauss–Legendre points are not enough when the B and C exponents are very different. A tight core 1s on B concentrates the integrand near s = 0.
   - For (H 1s H 1s | Cl 1s(ζ = 20.3) H 1s) in CH₃Cl the error at 48 points is 7 %, and it takes about 384 points to converge.
4. **M (and so h):** M is chosen for each radial integral by doubling it from 2 until two successive values agree to 10⁻¹⁴, rather than being fixed for each case. A few integrals have not settled to 10⁻¹⁴ by M = 256; the program prints a note when this happens (the largest change seen is about 10⁻¹²).

## Examples and reference values

The geometries use experimental bond lengths and angles, and the exponents come from the ADF DZP basis. The geometry code is in the `EXAMPLES` table of each file.

The reference values come from an independent code: an exact one-centre potential of the AA density, integrated over B·C on a fine molecular grid. They are converged to about 10⁻¹⁵.

The table shows the S + DE result minus the reference, for the paper's 48 outer points and for 384. Julia and Python agree to ~10⁻¹⁶.

| Case | Reference (hartree) | 48 points | 384 points |
|---|---|---|---|
| H₂O (O 1s O 1s \| H 1s H 1s), tight core on A | 0.27672300979425513 | 1.4e-13 | 2.2e-16 |
| H₂O (O 2pₓ O 2pₓ \| H 1s H 1s) | 0.11933172356252013 | 5.6e-14 | 1.4e-17 |
| H₂O (H 1s H 1s \| O 2pₓ H 1s), p on B | −0.081763823603047500 | −6.5e-15 | 4.2e-17 |
| H₂O (O 3d_z² O 2p_z \| H 1s H 2pₓ), d on A, p on C | 0.024153860413764196 | 2.5e-14 | 3.8e-17 |
| C₂H₄ (H 1s H 1s \| C 2pₓ C 2pₓ), p on B and C | 0.16865827648929535 | −4.7e-16 | −8.6e-16 |
| C₂H₄ (H 1s H 2p_z \| C 3d_xz C 2p_z), d on B, p on C | 0.0076285672783955706 | 2.6e-18 | 2.6e-18 |
| CO₂ (O 2p_z O 2p_z \| C 2p_z O 1s), moment expansion | −0.0084793229530729583 | −1.7e-18 | −1.4e-17 |
| CH₃Cl (H 1s H 1s \| Cl 1s H 1s), tight core on B | 0.0013544364600098090 | **9.7e-5** | 1.7e-18 |
| LiOH (Li 2p_z Li 2p_z \| O 1s H 1s), diffuse on A, tight on B | 0.013570501416502779 | **8.5e-7** | 1.1e-16 |

Run times:
- The whole example set at 384 points takes about 9 s in Julia (including compilation) and about 65 s in Python.
- Direct DE (`--method direct`) agrees with S + DE to ~10⁻¹⁶. It is faster in this implementation, because the S transformation's derivatives cost O(λ²) per node here.
