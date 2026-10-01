# LatticeBoltzmann.jl — models, equations, and how the kernels compute them

This document describes the library **as it is in `src/`**: a FluidX3D-style lattice Boltzmann solver for incompressible hydrodynamics, optional free-surface (FSLBM), optional temperature / melting, and melt-pool drivers (laser, powder). Every equation is tied to the function that implements it.

The module table of contents is `src/LatticeBoltzmann.jl`. Physics that is compiled in is selected in `src/extensions.jl`.

---

## 1. What the library is

The solver advances **one cell per GPU/CPU thread** through:

1. optional laser deposition into `Q` and powder into `msrc` / `mp`
2. FSLBM **surface_0** (mass streaming + gas reconstruction + CSF)
3. combined **stream-collide** (hydro D3Q19 + heat D3Q7 + forces)
4. FSLBM **surface_1 / 2 / 3** (interface conversion, excess mass)

The time loop is `step!` in `src/model.jl`. There is **no second population array**: streaming is the A-A pattern (even/odd slot swap) in `src/kernel.jl`.

Target continuum equations (lattice units unless noted):

| Physics | Continuum | When |
|---------|-----------|------|
| Hydro | Navier–Stokes, \(c_s^2 = 1/3\) | always (D3Q19) |
| Heat | advection–diffusion of enthalpy | `TEMPERATURE` |
| Melt | \(H = H_\text{sens}(T) + \Lambda f_\ell\) | `Λ > 0` |
| Free surface | mass-conserving FSLBM (Körner) | `SURFACE` |
| CSF | Young–Laplace via gas density | `σ ≠ 0` |
| Marangoni | \(\mathbf{F} = \sigma_T (\nabla T - \mathbf{n}(\mathbf{n}·\nabla T))\) | `σT ≠ 0` on `TYPE_I` |
| Buoyancy | Boussinesq | `β ≠ 0` |
| Darcy | Kozeny–Carman in mush | `K0 ≠ 0` |
| Evaporation | Hertz–Knudsen + Clausius–Clapeyron | `Λ_v > 0` |
| Recoil | Anisimov \(p_r = 0.54\, p_\text{sat}\) | `Λ_v > 0` |
| Radiation | \(\varepsilon\sigma_\text{SB}(T^4-T_\infty^4)\) | `C_rad > 0` |
| Laser | Gaussian bundle, PLIC + Fresnel | `model.laser` |
| Powder | ballistic Gaussian jet | `model.powder_jet` |

SI inputs are converted by `src/units.jl` in the `Model(Nx,Ny,Nz, units; …)` constructor (`src/model.jl`).

---

## 2. Compile-time switches

```1:16:src/extensions.jl
# Compile-time switches
const TRT                    = true # false evaluates as SRT
const SURFACE                = true
const VOLUME_FORCE           = true
const EQUILIBRIUM_BOUNDARIES = false
const MOVING_BOUNDARIES      = false
const FORCE_FIELD            = false
const TEMPERATURE            = true
const UPDATE_FIELDS          = SURFACE
const APPLY_FORCE            = VOLUME_FORCE || FORCE_FIELD || TEMPERATURE
```

These are **Julia `const`s**. `@static if SURFACE` / `TEMPERATURE` / `TRT` **delete whole fields and kernel branches at parse time**. A Domain built with `TEMPERATURE = false` has no `T`, `gi`, or `Q`.

| Switch | Effect |
|--------|--------|
| `TRT` | two-relaxation-time hydro (`collide_pair`); `false` → SRT/BGK |
| `SURFACE` | FSLBM fields + `surface_*` kernels + `stream_collide_surface_body!` |
| `TEMPERATURE` | D3Q7 heat, enthalpy, melt, evap, radiation |
| `VOLUME_FORCE` | Guo forcing from `fx,fy,fz` (gravity) |
| `FORCE_FIELD` | extra per-cell `F[n,:]` |
| `EQUILIBRIUM_BOUNDARIES` | `TYPE_E` cells overwrite \(f_i = f_i^\text{eq}\) |
| `MOVING_BOUNDARIES` | bounce-back with wall velocity (`TYPE_MS`) |
| `UPDATE_FIELDS` | write `ρ,u` during collide (on when `SURFACE`) |
| `APPLY_FORCE` | Guo term + half-force shift of `u` |

`SURFACE` without `VOLUME_FORCE` warns: gravity is ignored.

---

## 3. Lattice units (`src/units.jl`)

A `Units` object stores five scales: cell size `m` (metres), mass `kg`, time `s`, temperature `K` (kelvin per lattice \(T\)), and `cp` [J/(kg·K)].

Typical construction: pick lattice Mach `u`, match a physical length and speed:

```14:19:src/units.jl
function Units(x, u, ρ, si_x, si_u, si_ρ; T::Type{<:AbstractFloat}=Float32, K=1, cp=1)
    m  = T(si_x / x)
    kg = T(si_ρ / ρ) * m^3
    s  = T(u / si_u) * m
    Units{T}(m, kg, s, T(K), T(_cp_si(cp)))
end
```

For melt pools, `K` is usually the melting temperature so lattice \(T \sim O(1)\).

Hydro identities used everywhere:

- \(c_s^2 = 1/3\) on D3Q19 → `si_p = ρ_si · (m/s)² / 3`
- viscosity: \(\tau = 3\nu + 1/2\), \(\omega = 1/\tau\) (`omega_from_nu` in `kernel.jl`)
- Model thermal diffusivity \(\alpha\) is **twice** the Chapman–Enskog \(\chi\) of D3Q7 (see §8)

Helpers: `lbm_ν`, `lbm_σ`, `lbm_g`, `lbm_Q`, `lbm_Λ`, `lbm_rad`, `lbm_evap`, …

---

## 4. Architecture

### 4.1 Memory

`src/memory.jl`. Populations are **direction-major**:

```
f_index(n, i, N) = n + (i-1)*N
```

All cells of \(f_1\), then all of \(f_2\), … Neighbor `n` and `n+1` are adjacent in memory → coalesced GPU loads along the cell index.

`Memory` wraps a host or device array. `MemoryContainer` maps a **global** cell `n` onto one or more subdomain buffers (`Dx,Dy,Dz`). One box: `Dx=Dy=Dz=1`.

### 4.2 Domain

`src/domain.jl` — one grid’s state: `fi` (D3Q19), `ρ`, `u`, `F`, `flags`, plus FSLBM / T fields when compiled in. `t::UInt64` is the step counter (even/odd selects the AA kernel).

### 4.3 Model

`src/model.jl` — `domains`, views (`model.ρ`, `model.u`, …), **cached** KernelAbstractions kernels (compiled once), `units`, optional `laser` / `powder_jet`.

`run!(model, nsteps)` → `initialize!` once, then `step!` each step.

---

## 5. Cell types (`src/flags.jl`)

Each flag is **one bit** of `UInt8` so types can be OR’d.

| Hex | Name | Meaning |
|-----|------|---------|
| `0x01` | `TYPE_S` | solid, bounce-back |
| `0x02` | `TYPE_E` | equilibrium I/O |
| `0x04` | `TYPE_T` | Dirichlet temperature (on a solid) |
| `0x08` | `TYPE_F` | fluid |
| `0x10` | `TYPE_I` | interface |
| `0x20` | `TYPE_G` | gas (no hydro collide) |
| `0x40` | `TYPE_H` | heat-flux / Robin wall |

Masks / one-step FSLBM transitions (same bits OR’d):

| Mask | Bits | Role |
|------|------|------|
| `TYPE_BO` | `S\|E` | “is this a wall-like cell?” |
| `TYPE_MS` | `S\|E` | moving solid (when `MOVING_BOUNDARIES`) |
| `TYPE_IF` | `I\|F` | interface just filled → become **F** |
| `TYPE_IG` | `I\|G` | interface emptied → become **G** |
| `TYPE_GI` | `F\|I\|G` | gas gained mass → become **I** |
| `TYPE_SU` | `F\|I\|G` | extract free-surface bits (`== TYPE_GI` numerically) |

`surface_3` commits `IF→F`, `IG→G`, `GI→I`.

Fill fraction (`calculate_phi` in `kernel.jl`):

\[
\phi =
\begin{cases}
1 & \text{TYPE\_F} \\
\mathrm{clamp}(m/\rho,0,1) & \text{TYPE\_I} \\
0 & \text{else (gas)}
\end{cases}
\]

---

## 6. Hydrodynamics

### 6.1 Discrete velocity set

Default scheme `:D3Q19` (`src/velocities.jl`, `src/weights.jl`). Velocities are stored as **± pairs** after rest: \(e_0, +e_1,-e_1,+e_2,-e_2,\ldots\) That pairing is required for AA and TRT.

Weights: \(w_0=1/3\), face \(1/18\), edge \(1/36\).

### 6.2 Equilibrium (recovering NS)

```49:52:src/kernel.jl
@inline function feq(wi, ρn, ux, uy, uz, uu, ci, ::Type{CType}) where {CType}
    cu = CType(ci[1])*ux + CType(ci[2])*uy + CType(ci[3])*uz
    return wi * ρn * (one(CType) + CType(3)*cu + CType(4.5)*cu*cu - uu)
end
```

with `uu = 1.5 (u·u)`. This is the standard second-order expansion at \(c_s^2=1/3\):

\[
f_i^{\mathrm{eq}} = w_i\rho\bigl(1 + 3\,c_i·u + \tfrac{9}{2}(c_i·u)^2 - \tfrac{3}{2}u^2\bigr).
\]

Moments: \(\rho = \sum f_i\), \(\rho u = \sum c_i f_i\) (plus Guo’s \(\Delta t F/2\), below). Chapman–Enskog → incompressible NS with \(\nu = c_s^2(\tau^+-1/2) = (\tau^+-1/2)/3\).

### 6.3 SRT vs TRT

**SRT / BGK** (`TRT = false`):

\[
f_i^* = (1-\omega) f_i + \omega f_i^{\mathrm{eq}}, \quad \omega = 1/(3\nu+1/2).
\]

**TRT** (`collide_pair`, default on): split each ± pair into even/odd moments

\[
f^\pm = \tfrac12(f_+ \pm f_-),\quad
f_\pm^* = f_\pm + \tfrac12 \omega^+ (e^\text{sum}-f^\text{sum}) \pm \tfrac12 \omega^- (e^\text{dif}-f^\text{dif}).
\]

\(\omega^+ = \omega\) from viscosity. The **magic parameter** \(\Lambda = 3/16 = 0.1875\):

```89:92:src/kernel.jl
@inline function omega_minus(ω::CType) where {CType}
    three_nu = one(CType) / ω - CType(0.5) # 3ν = τ⁺ − 1/2
    return one(CType) / (CType(0.1875) / three_nu + CType(0.5))
end
```

\[
\tau^- = \frac{\Lambda}{\tau^+-1/2} + \tfrac12, \quad \omega^- = 1/\tau^-.
\]

This cancels a large class of bounce-back error (Ginzburg / TRT literature). \(\omega\) is clamped to \((0.05, 1.95)\) (`clamp_omega`) so BGK never sits on \(\omega=0\) or \(2\).

### 6.4 Guo forcing

Body force \(\mathbf{F}\) (gravity, Marangoni, recoil, Darcy, Boussinesq, optional `F` field) enters two places:

1. **Shifted velocity** used in \(f^{\mathrm{eq}}\) and in `u` stored for output:

\[
\mathbf{u} = \frac{1}{\rho}\sum_i c_i f_i + \frac{\Delta t}{2\rho}\mathbf{F}.
\]

(`APPLY_FORCE` block in `stream_collide_*_body!`.)

2. **Source in collide** (`guo_fi`):

\[
F_i = 9 w_i \Bigl[(c_i·F)(c_i·u + \tfrac13) - \tfrac13 u·F\Bigr],
\]

then scaled by \((1-\omega/2)\) (SRT) or the TRT even/odd split (`scale_force_pair`). Rest population: `guo_rest`. Pairs: `guo_pair`.

`prescribed_hydro` clamps \(|u|\) to \(c_s\) so Mach stays < 1.

### 6.5 Bounce-back and walls

`TYPE_S` **does not collide**. AA already stores the incoming population in the opposite slot: that *is* bounce-back.

Moving walls (`MOVING_BOUNDARIES`): `moving_wall_pair` adds the Ladd term \(-6 w_i \rho_w (c_i·u_w)\) when the neighbor is `TYPE_S` and this cell is `TYPE_MS`.

Equilibrium I/O (`TYPE_E`): `equilibrium_boundary!` overwrites all \(f_i\) with \(f_i^{\mathrm{eq}}(\rho,u)\).

Periodic wrap: `wrap_coord` / `src_index` — faces wrap with a branch-light `ifelse` (no `%`).

---

## 7. How hydro is computed efficiently (A-A + kernels)

### 7.1 A-A streaming (no second `f` array)

Velocities come in ± pairs. After collide, the post-collision \(f_+\) is written into the slot where the **next** collide will read the streamed value.

```16:29:src/kernel.jl
@inline load_pair(fi, n, src, i, ::Val{true}, N, ::Type{CType}) where {CType} =
    (CType(fi[f_index(n, i, N)]), CType(fi[f_index(src, i + 1, N)]))
@inline load_pair(fi, n, src, i, ::Val{false}, N, ::Type{CType}) where {CType} =
    (CType(fi[f_index(n, i + 1, N)]), CType(fi[f_index(src, i, N)]))
```

- **Even** step (`Val{false}`): load \((f_{i+1}(n),\, f_i(\mathrm{src}))\).
- **Odd** step (`Val{true}`): load \((f_i(n),\, f_{i+1}(\mathrm{src}))\).

`store_pair!` writes the collided pair into those same slots. `src` is the neighbor along \(+c_i\). One array, no halo copy of `fi`, stream and collide are **one kernel**.

`domain.t` even → `stream_collide_even_kernel!`; odd → `*_odd_kernel!`. The `Val{odd}` is a **compile-time** bool so the two layouts inline without a runtime branch in the inner pair loop.

### 7.2 One thread per cell

```1367:1379:src/kernel.jl
@kernel function stream_collide_even_kernel!(...)
    n = @index(Global)
    @inbounds stream_collide_body!(Val(false), ..., Int(n), ...)
end
```

KernelAbstractions launches `ndrange = N`. The same source runs on `CPU()` and `CUDABackend()`.

### 7.3 D3Q19 specialization

There are **two** `stream_collide_body!` overloads when `SURFACE` is off:

- generic: `ntuple(Val(NP))` loop over pairs (any Q)
- `NTuple{19,…}`: **unrolled** `src2…src18` and named `fp2,fm3,…` — no loop-carried pair array, better GPU occupancy

With `SURFACE`, `stream_collide_surface_body!` uses the generic pair `ntuple` (Q is still 19 in practice).

### 7.4 Dead-strip and `@inbounds`

`@static if TEMPERATURE` around `collide_temperature!` means a hydro-only build never emits heat registers. `@inbounds` on the kernel body skips bounds checks.

### 7.5 Property blending

Local \(\nu(T,f_s)\) and \(\alpha(T,f_s)\):

```
p = p_s + p_{sT}(T-T_\text{avg})   (and liquid analog)
p = clamp to ≥ 10% of |p_0|
p = f_s p_s + (1-f_s) p_ℓ
ω = 1/(3ν+1/2) or 1/(2α+1/2)
```

(`prop_fs_T`, `omega_from_nu`, `omega_T_from_alpha`.) So mushy cells have intermediate viscosity/conductivity without a second lattice.

---

## 8. Temperature (D3Q7, Peng)

Enabled by `TEMPERATURE`. Comment at the top of the thermal block:

```205:208:src/kernel.jl
# D3Q7 thermal. Perturbation g' = g - w, T = 1 + Σg.
# ω_T = 1/(2α + 1/2). Boussinesq: F_eff = F - F β (T - T_avg).
# Volumetric Q: after SRT, add Δgeq(ΔT=Q) so ΣΔg = Q (lattice dT/step).
```

### 8.1 Populations and CE diffusivity

D3Q7: rest + six axes (`weights.jl`: \(w_0=1/4\), \(w_{\mathrm{axis}}=1/8\)). Thermal speed of sound \(c_{sT}^2 = 1/4\).

Macroscopic temperature:

\[
T = 1 + \sum_{i=0}^{6} g_i
\]

(`Tfromg` in `collide_temperature!`). Equilibrium (shift so \(g=0\) means \(T=1\)):

\[
g_0^{\mathrm{eq}} = \tfrac14 T - \tfrac14, \qquad
g_{\pm\alpha}^{\mathrm{eq}} = \tfrac12 T\, u_\alpha + \tfrac18(T-1).
\]

(`geq_T_rest`, `geq_T_axis`.)

Relaxation:

\[
\omega_T = \frac{1}{2\alpha + 1/2} \quad\Rightarrow\quad \tau_T - \tfrac12 = 2\alpha.
\]

Chapman–Enskog on this stencil:

\[
\chi = c_{sT}^2(\tau_T-\tfrac12) = \tfrac14 · 2\alpha = \tfrac\alpha2.
\]

So Domain **`α` is twice** the CE diffusivity (and `thermal_conductivity(ω_T) = 0.25(1/ω_T - 1/2) = α/2`). `lbm_αT` in `units.jl` uses Model-\(\alpha = 2k/(\rho c_p)\).

Heat collide is **SRT** (not TRT). After collide, a source \(\Delta T = Q\) is injected as \(\Delta g^{\mathrm{eq}}(\Delta T=Q)\):

\[
\sum_i \Delta g_i = Q
\]

so \(T\) jumps by \(Q\) this step. Explicitly (`collide_temperature!` else-branch): rest gets \(0.25 Q\), each axis \(0.5 Q u_\alpha + 0.125 Q\).

AA for `gi` uses the **same** `load_pair` / `store_pair!` as hydro (only three axis pairs).

### 8.2 Enthalpy and melting

Sensible heat with optional \(c_p(T)\):

\[
\frac{c_p}{c_{p,\mathrm{ref}}} = 1 + \gamma (T-1), \quad
H_\text{sens} = \int_0^T (1+\gamma(\theta-1))\,d\theta = T + \gamma(\tfrac12 T^2 - T).
\]

(`sensible_H`; \(\gamma=0\) → \(H=T\).) \(\gamma_s,\gamma_l\) blend with \(f_s\).

Latent heat \(\Lambda = L/(c_p K)\) (lattice temperature units, `lbm_Λ`):

\[
H = H_\text{sens}(T,\gamma) + \Lambda f_\ell, \quad f_\ell = 1-f_s.
\]

`invert_enthalpy` solves \(H \mapsto (T, f_\ell)\) in three regions: solid \(H \le H(T_s)\), mush \(T\in[T_s,T_l]\), liquid \(H \ge H(T_l)+\Lambda\). Equal \(T_s=T_l\) is isothermal Stefan.

In collide, if \(\Lambda\neq 0\) or \(\gamma\neq 0\):

1. \(H \leftarrow H(T_\text{from }g, f_s) + Q\)
2. \((T,f_\ell) = \mathrm{invert}(H)\)
3. mush: write \(g=g^{\mathrm{eq}}(T)\) (`write_geq`); fully solid/liquid: keep SRT and set \(Q \leftarrow T_\text{new}-T_\text{from }g\)

Solid fraction from temperature only (Dirichlet walls): `fs_from_T`.

### 8.3 Thermal BCs

| Flag | Model | Code |
|------|--------|------|
| `TYPE_S` only | adiabatic bounce-back (AA, do not overwrite `g`) | comment on `reconstruct_g_boundaries!` |
| `TYPE_S\|TYPE_T` | Dirichlet ABB at `T[wall]` | `is_dirichlet_solid`; \(g_{\mathrm{in}} = 2g^{\mathrm{eq}}(T_w)-g_{\mathrm{out}}\) |
| `TYPE_S\|TYPE_H` | Neumann / Robin | `robin_wall_T` |

Robin (Fourier + convection), \(k = \alpha/2\):

\[
T_w = \frac{T_\text{fluid} + q/k + \mathrm{Bi}\, T_\infty}{1+\mathrm{Bi}}, \quad \mathrm{Bi}=h/k.
\]

`h=0` → Neumann with flux `Q[wall]`. `T[wall]` holds \(T_\infty\) when `h≠0`.

`reconstruct_g_boundaries!` runs in **surface_0** on F and I cells so a missing wall–fluid link (AA only streams \(+c\)) is rebuilt. Wall enthalpy change goes to `Eacc[EACC_WALL]`.

A `TYPE_G` face is adiabatic at the interface temperature: `surface_0` sets \(g_{\mathrm{in}} = 2g^{\mathrm{eq}}(T)-g_{\mathrm{out}}\). Copying \(g_{\mathrm{out}}\) alone is a sink. Do not `store_pair` from the gas cell into the melt; that overwrites the interface equilibrium. Gas init writes equilibrium only into the gas cell's own slots.

### 8.4 Boussinesq

After thermal collide, `collide_temperature!` returns a modified force:

\[
\mathbf{F} \leftarrow \mathbf{F} - \mathbf{F}_0\,\beta\,(T-T_\text{avg})
\]

i.e. gravity is scaled by density variation \(\beta\Delta T\). \(\mathbf{F}_0=(f_x,f_y,f_z)\).

---

## 9. Free-surface LBM (Körner / FluidX3D)

`SURFACE`. Liquid is `TYPE_F` / `TYPE_I`; atmosphere is `TYPE_G` (no hydro collide). Interface is **one cell** thick.

### 9.1 Time splitting (`step!`)

```
deposit_laser!, advance_powder_jet!, powder_gas_kernel!   # drivers
surface_0     # mass flux + gas f reconstruction + CSF ρ_g + g BCs
stream_collide_surface_body!   # hydro+heat collide (F and I only)
surface_1     # IF neighbors: G→GI, IG→I
surface_2     # GI: init feq from liquid neighbors; IG: F→I
surface_3     # commit IF/IG/GI; excess mass to neighbors
```

### 9.2 Mass

Fluid cells: \(\Delta m = \sum_i (f_i^{\mathrm{in}}-f_i^{\mathrm{out}})\) plus excess from neighbors (`surface_0`).

Interface: flux to a **fluid** neighbor is the full pop difference; to another **interface**, it is weighted \(\frac12(\phi_j+\phi_n)\) (Körner).

`surface_3`: F keeps \(m=\rho\) and dumps \(m-\rho\) as `massex`; I clamps \(m\in[0,\rho]\); excess is split equally among liquid neighbors.

Empty I (`m<0` or no F neighbor) → `TYPE_IG`. Full I (`m>ρ` or no G neighbor) → `TYPE_IF`. Frozen solid (`f_s ≈ 1`) does not convert.

### 9.3 Gas reconstruction and capillary pressure

Missing populations from gas are filled with the Körner rule (`surface_0`):

\[
f_i = f_i^{\mathrm{eq}}(\rho_g,u_g) + f_{\bar i}^{\mathrm{eq}}(\rho_g,u_g) - f_{\bar i}.
\]

Gas density from **PLIC curvature** (`gas_density_plic` in `plic.jl`):

\[
\rho_g = 1 - 6\sigma\kappa.
\]

With \(p=\rho/3\), this is Young–Laplace \(\Delta p \propto \sigma\kappa\) (FluidX3D convention). \(\sigma=0\) → \(\rho_g=1\).

\(\sigma(T)=\sigma+\sigma_T(T-T_\sigma)\) on the interface before the PLIC call.

\(u_g\) is the liquid velocity plus half-force (gravity, Marangoni, recoil) so Guo consistency holds at the free surface.

### 9.4 PLIC (`src/plic.jl`)

- **Parker–Youngs** normal `calculate_normal_py` on the D3Q27 \(\phi\) stencil (metal → gas).
- **`plic_cube(V, n)`**: plane offset in the unit cube for fill \(V\) and normal \(n\) (analytic cases + trig for the rest).
- **`calculate_curvature`**: fit a paraboloid \(z = A x^2 + B y^2 + C xy + Hx + Iy\) to neighboring PLIC intercepts; mean curvature from the second fundamental form; clamp \(\kappa\in[-1,1]\).

Used for CSF (`ρ_g`) and for **laser hits** (same plane).

---

## 10. Other continuum forces (all Guo-forced)

Computed in `stream_collide_surface_body!` (and the no-surface body) **after** moments, **before** collide.

### 10.1 Darcy / Kozeny–Carman (`darcy_force`)

\[
K = K_0 \frac{f_\ell^3}{f_s^2+\varepsilon},\quad
\mathbf{F}_D = -\frac{\nu}{K}\mathbf{u}.
\]

Fully solid (\(f_\ell < 10^{-3}\)): force **replaces** all other forces. Explicit Guo with equilibrium at \(u+F/(2\rho)\) maps the population velocity to \(u'=(1-\mathrm{drag}/\rho)\,u\), so the cap is \(\mathrm{drag}=\rho\) (one-step kill). \(\mathrm{drag}=2\rho\) would store a zero velocity and reflect the populations. `K0=0` → off. `K0` is lattice length² (`Model` divides SI m² by `units.m²`).

### 10.2 Marangoni (`marangoni_force`)

On `TYPE_I` if `σT ≠ 0` and not frozen:

\[
\mathbf{n} = \nabla\phi/|\nabla\phi|, \quad
\mathbf{F} = \sigma_T \bigl(\nabla T - \mathbf{n}(\mathbf{n}·\nabla T)\bigr).
\]

Central differences; one-sided if a neighbor is not liquid (`axis_deriv`). Tangential \(\nabla T\) only.

### 10.3 Recoil (`recoil_force`)

Anisimov:

\[
p_r = 0.54\, p_{\mathrm{sat}}(T),\quad
\mathbf{F} = p_r\,\mathbf{n},\quad \mathbf{n}=\nabla\phi/|\nabla\phi|
\]

into the liquid. \(p_r\) capped at \(0.5\) so Guo \(\Delta u\) stays \(O(0.1)\). Off if `Λ_v=0`.

---

## 11. Evaporation and radiation

Only on **`TYPE_I`**, not Dirichlet/flux walls, not frozen (`collide_temperature!`).

### 11.1 Clausius–Clapeyron + Hertz–Knudsen

\[
p_{\mathrm{sat}} = p_{0v}\,\exp\bigl[-\beta_v(1/T-1/T_v)\bigr],
\quad
\dot m = C_{\mathrm{hk}}\, p_{\mathrm{sat}} / \sqrt{T},
\quad
Q_e = \dot m\,\Lambda_v.
\]

`p_sat`, `evaporative_dT`, `evaporative_flux`. \(Q_e\) is lattice \(\Delta T\) this step, capped so \(T\) cannot fall below \(T_v\) in one collide. Mass removed: `mass -= ṁ ρ` (`MACC_EVAP`).

`lbm_evap` sets \(\Lambda_v=L_v/(c_p K)\), \(\beta_v=L_v/(R_{\mathrm{sp}} K)\), lattice \(p_0\), and

\[
C_{\mathrm{hk}} = \frac{m}{s\sqrt{2\pi R_{\mathrm{sp}} K}}.
\]

### 11.2 Radiation

\[
Q_{\mathrm{rad}} = C_{\mathrm{rad}}(T^4 - T_{\mathrm{rad}}^4),
\quad
C_{\mathrm{rad}} = \frac{\varepsilon\sigma_{\mathrm{SB}} K^3 s}{\rho c_p m}.
\]

`lbm_rad` / `radiation_dT`. Cooling only (\(T>T_\infty\)); capped to not cross \(T_\infty\) in one step.

---

## 12. Laser (`src/laser.jl`)

Not a lattice kernel: **host** ray march each `step!` (copied flags/ϕ).

- Bundle: `nrays×nrays` samples of a Gaussian \(I = I_0 \exp(-2r^2/w^2)\) with \(I_0 = 2P/(\pi w^2)\), renormalized to power `P` [W].
- DDA walk through cells. On `TYPE_I`, **PLIC hit** (`plic_hit`): plane \(\mathbf{n}·(\mathbf{r}-\mathbf{c}) = \texttt{plic_cube}(\phi,\mathbf{n})\).
- Front face: \(-\hat d·n > 0\). **Unpolarized Fresnel** absorptance vacuum→\(\tilde n = n+ik\) (`fresnel_absorptance`).
- Absorbed \(P_{\mathrm{abs}}\) is converted to lattice \(Q\) by `laser_qfac = s / (ρ cp K m³)` (joules this step → \(\Delta T\) in one cell).
- Optional `skin`: spread the hit along \(-\mathbf{n}\) into metal (`_deposit_along_normal!`).
- Remainder reflects (`max_bounce`). Hitting `TYPE_F` dumps remaining power. `TYPE_S` stops the ray.

`Q` is **zeroed then rewritten** each deposit (`every` steps). Thermal collide treats `Q` as volumetric \(\Delta T\).

---

## 13. Powder (`src/powder.jl` + `powder_gas_kernel!`)

`PowderJet`: Gaussian in the nozzle plane, ballistic parcels at speed `v`, mass rate `mdot` [kg/s] split across `nparcels` per step. `advance_powder_jet!` deposits lattice mass into `msrc` on `TYPE_I`/`F`.

In collide (`τ_p > 0`):

- add `msrc·ρ` into `mp` (unmelted)
- if predicted \(T \ge T_s\): melt up to one cell of `mp` into `mass`, debit latent+sensible from `Q`
- else: `mp *= exp(-1/τ_p)` (lifetime decay)

`τ_p = 0`: `msrc` goes straight into `mass` (skip if frozen). Gas cells never become metal: `powder_gas_kernel!` only decays `mp` on `TYPE_G`.

---

## 14. Conservation ledgers

Not dynamics — diagnostics.

Energy (`Eacc`, 5 slots in `domain.jl`):

\[
H \approx H_0 + Q_{\mathrm{laser}} - Q_{\mathrm{rad}} - Q_{\mathrm{evap}} + H_{\mathrm{powder}} - Q_{\mathrm{wall}}.
\]

Mass (with `SURFACE`):

\[
M \approx M_0 + m_{\mathrm{powder}} - m_{\mathrm{evap}}.
\]

`acc_add!` uses `Atomix.@atomic` so many cells can increment the same bin on GPU. `energy_budget` / `mass_budget` report `residual = actual - expected` (streaming/BC leak).

Enthalpy of a cell: `fill · cell_enthalpy(T, f_s, Λ, γ)` with `fill=ϕ` on the free surface.

---

## 15. Initialization

`initialize_kernel!` → `initialize_body!` (`SURFACE`) or the no-surface kernel.

- Bare cells (no S/E/T/H/F/I) become **G**.
- G next to F becomes **I** with \(\phi=0.5\) and \(\rho,u\) averaged from fluid neighbors.
- Solids: \(u=0\), `store_geq_local!(T_{\mathrm{wall}})`.
- F/I: `store_feq!` / `store_geq!` in the **even** AA layout (`Val(false)`), `mass = ϕ ρ`.

Then `reset_energy_budget!` / `reset_mass_budget!` snapshot \(H_0\), \(M_0\).

---

## 16. File map

| File | Role |
|------|------|
| `extensions.jl` | compile-time physics |
| `flags.jl` | cell-type bits |
| `weights.jl` / `velocities.jl` | D3Q19 / D3Q7 stencils |
| `units.jl` | SI ↔ lattice |
| `memory.jl` | DDF layout, host/device |
| `domain.jl` | state + energy/mass budgets |
| `kernel.jl` | collide, heat, FSLBM 1–3, powder-on-gas, moments |
| `model.jl` | `run!`/`step!`, **surface_0**, VTK `export!` |
| `plic.jl` | normal, cube plane, curvature, \(\rho_g\) |
| `laser.jl` | Gaussian + Fresnel + PLIC rays |
| `powder.jl` | ballistic jet |
| `log.jl` | run log files |

`test/` mirrors these (hydro, T, PLIC, force, moving walls). `examples/` turn subsets on: lid-driven cavity (`MOVING_BOUNDARIES`), dam break (FSLBM), Stefan / freeze layer (melt), thermocapillary (Marangoni), melt_pool (laser+powder+FS+T).

---

## 17. What is *not* in this tree

`src/` has **no** dissolved-gas / bubble-tracker / foam module. Free-surface gas is a **passive atmosphere** (`TYPE_G`), not a \(pV=nRT\) bubble lattice. Foam/Henry/nucleation work, if any, lives outside this `src/` snapshot.

---

## 18. Suggested reading order in the code

1. `extensions.jl`, `flags.jl`
2. `feq`, `collide_pair`, `load_pair` / `store_pair!` (`kernel.jl` top)
3. `stream_collide_body!` D3Q19 overload (moments → Guo → collide → store)
4. `collide_temperature!`
5. `step!` then `surface_0_body!` → `surface_1/2/3`
6. `plic.jl` `gas_density_plic`
7. `laser.jl` `_walk_laser_ray!`, `powder.jl` `PowderJet`
