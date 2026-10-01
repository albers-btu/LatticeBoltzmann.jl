# Solid 316L / TiH2 compact, foamed by a laser. Requires FOAM, SURFACE,
# TEMPERATURE. n_hydro stays 1. This is not the pre-seeded aluminum melt
# in foam_poisson.jl: the charge starts solid, with no pores and no
# dissolved gas.
#
# The compact is 80% 316L and 20% TiH2 by volume. A cell is 0.17 mm, much
# larger than a hydride particle, so each metal cell carries a mixture.
# The local TiH2 fraction varies about that 20% mean (seed 1); richer
# cells are the ones that can nucleate. The matrix is one material:
# 316L density, conductivity, latent heat, solidus, liquidus.
# TiH2 is a volume-fraction field. It is copied into `mp` for ParaView
# (powder lifetime is zero, so the solver does not treat `mp` as powder).
# Colour `mp` for the hydride, `T` for the spot, `fs` for solid fraction
# (0 liquid, 1 solid), `c` for dissolved hydrogen, `tag` for pores.
# Contour phi = 0.5. ParaView 6: do not isosurface a flat array.
#
# TiH2 → Ti + H2. Untreated hydride releases near 400°C, while 316L is
# still solid, so this charge uses a pretreated hydride that starts at
# 900 K. The rate is quadratic in (T − 900 K) and saturates at the
# liquidus, so most of the particle is still there when the matrix melts.
# Each unit of hydride volume fraction releases `c_yield` of dissolved
# concentration into that cell. A full conversion of the 20% loading is
# a lattice inventory, not the ~10³ gas volumes of H2 at 1 bar: the pores
# grow from V_m * c by the same law as the aluminum example,
# d(R²)/dt = 2 c V_m D.
#
# A nucleus is punched only in a molten cell (T at or above the liquidus,
# solid fraction below 0.1) whose initial hydride fraction is at least the
# mean and whose dissolved hydrogen is above c_nuc. Release consumes the
# local hydride, so the abundance gate uses the loading from t = 0. The
# score is that loading × dissolved hydrogen × superheat, so a hotter,
# richer site wins. At most one pore
# per step, centers at least rmin apart. rmin = 2 R + 4, so the film
# starts outside the disjoining range. The punch radius is 1.5 cells so
# a nucleus fits in the top layers the beam melts; the hydrogen inflates
# it after that.
#
# Liquid 316L is about 5×10⁻³ Pa·s. On this grid that viscosity cannot
# carry both gravity and surface tension with n_hydro = 1, so the melt
# uses the same stabilized viscosity as the aluminum foam (ν_lat = 0.2,
# which is Pr = 40 against the real conductivity). σ is the 1.6 N/m of
# 316L. The cell size puts g = 9.81 m/s² at |g_lat| = 2×10⁻⁵.
# The heat uses Pr = 2. At Pr = 40 the beam heats one cell and that
# cell's temperature climbs without bound, because the power cannot
# conduct away. Pr = 2 makes a pool a few cells deep.
#
# Open output_foam_tih2/lbm.pvd.
# FOAM_TIH2_STEPS overrides the step count.

using LatticeBoltzmann
using Printf
using Random
using Unitful

@assert FOAM && SURFACE && TEMPERATURE

# --- 316L matrix, pretreated TiH2 ---
si_ρ     = 8000u"kg/m^3"
si_cp    = 500u"J/kg/K"
si_k     = 30u"W/m/K"
si_Ts    = 1648u"K"
si_Tl    = 1673u"K"
si_Tinit = 300u"K"
si_Tdec  = 900u"K"
si_L     = 2.8e5u"J/kg"
si_σ     = 1.6u"N/m"
si_g     = 9.81u"m/s^2"
si_K0    = 1.0e-10u"m^2"
si_P     = 18000u"W"
si_w     = 1.2e-3u"m"          # 1/e² radius

α_si = si_k / (si_ρ * si_cp)
ν_lat = 0.2
Pr_ν = 40
Pr_heat = 2
si_ν = Pr_ν * α_si
ν_si = ustrip(u"m^2/s", si_ν)
g_si = ustrip(u"m/s^2", si_g)
g_lat_target = 2.0e-5
m = cbrt(g_lat_target * ν_si^2 / (g_si * ν_lat^2))
s = ν_lat * m^2 / ν_si
lbm_u = 0.05
si_u = (lbm_u * m / s) * u"m/s"
units = Units(m * u"m", si_u, si_ρ; x=1, u=lbm_u, ρ=1, T=Float32,
              K=ustrip(u"K", si_Tl), cp=si_cp)

ν_l = Float64(lbm_ν(units, si_ν))
σ_l = Float64(lbm_σ(units, si_σ))
g_l = Float64(lbm_g(units, -si_g))
α_ce = ν_lat / Pr_heat
@assert 0.19 < ν_l < 0.21
@assert 0.001 < σ_l < 0.06
@assert -3.0e-5 < g_l < -1.2e-5

# 8 mm × 6 mm × 3 mm compact, walls on every face, gas headspace above.
si_Lx = 8.0e-3
si_Ly = 6.0e-3
si_Hz = 3.0e-3
Nx = max(24, round(Int, si_Lx / m) + 2)
Ny = max(24, round(Int, si_Ly / m) + 2)
H = max(14, round(Int, si_Hz / m) + 1)
n_head = 16
Nz = H + n_head + 1

# Hydrogen release and nucleation. Lattice units.
const TIH2_FRACTION = 0.20
const C_YIELD = 0.25f0          # dissolved c from one full hydride cell
const K_REL = 0.005f0           # fraction of remaining hydride per step at the liquidus
const C_NUC = 0.008f0
const X_NUC = 0.20f0            # initial loading at or above the 20% mean
const FS_LIQUID = 0.1f0         # solid fraction; below this the matrix is molten
const R_NUC = 1.5
const RMIN_NUC = 2 * R_NUC + 4  # film of 4 cells, disjoining starts at 0
const N_NUC_MAX = 8
const LASER_ON = 50

D = 0.015
V_m = 8.0
k_H = 1e-5
k_Π = 0.015
seed = 1
dir = "output_foam_tih2"
fields = (:phi, :c, :tag, :rho, :u, :flags, :T, :fs, :mp)
nsteps = something(tryparse(Int, get(ENV, "FOAM_TIH2_STEPS", "")), 2500)
every = nsteps > 400 ? 100 : max(1, nsteps ÷ 8)

function compact_flags(Nx, Ny, Nz, H)
    flags = fill(TYPE_G, Nx * Ny * Nz)
    for z in 0:(Nz - 1), y in 0:(Ny - 1), x in 0:(Nx - 1)
        n = 1 + x + Nx * y + Nx * Ny * z
        if x == 0 || y == 0 || z == 0 || x == Nx - 1 || y == Ny - 1 || z == Nz - 1
            flags[n] = TYPE_S | TYPE_T
        elseif z < H
            flags[n] = TYPE_F
        end
    end
    return flags
end

# Continuous loading. Raw values are 0.25–1.25, then rescaled so the
# compact mean is `fraction` (about 0.07–0.33).
function paint_tih2(flags; fraction=TIH2_FRACTION, seed=1)
    rng = Xoshiro(seed)
    x = zeros(Float32, length(flags))
    nmet = 0
    for n in eachindex(flags)
        (flags[n] & TYPE_SU) == TYPE_F || continue
        x[n] = 0.25f0 + rand(rng, Float32)
        nmet += 1
    end
    s = sum(x)
    scale = nmet == 0 || s == 0 ? 0f0 : Float32(fraction * nmet / s)
    @inbounds for n in eachindex(x)
        x[n] *= scale
    end
    return x
end

function tih2_mean(tih2, flags)
    s = 0.0
    n = 0
    for i in eachindex(flags)
        su = flags[i] & TYPE_SU
        (su == TYPE_F || su == TYPE_I) || continue
        s += Float64(tih2[i])
        n += 1
    end
    n == 0 && return 0.0
    return s / n
end

# ξ = 0 below T_dec, 1 at the liquidus, quadratic in between.
function release_hydrogen!(tih2, T, flags, T_dec, Tl, k_rel, c_yield)
    δ = zeros(Float32, length(tih2))
    span = max(Tl - T_dec, 1f-6)
    @inbounds for n in eachindex(tih2)
        tih2[n] <= 0 && continue
        su = flags[n] & TYPE_SU
        (su == TYPE_F || su == TYPE_I) || continue
        Tn = T[n]
        Tn <= T_dec && continue
        ξ = Tn >= Tl ? 1f0 : ((Tn - T_dec) / span)^2
        Δx = tih2[n] * k_rel * ξ
        tih2[n] -= Δx
        δ[n] = c_yield * Δx
    end
    return δ
end

function _xyz(n, Nx, Ny)
    n0 = n - 1
    x = n0 % Nx
    y = (n0 ÷ Nx) % Ny
    z = n0 ÷ (Nx * Ny)
    return x, y, z
end

function near_pore(tags, cx, cy, cz, Nx, Ny, rmin)
    r2 = rmin * rmin
    for n in eachindex(tags)
        tags[n] > 0 || continue
        x, y, z = _xyz(n, Nx, Ny)
        dx = x + 0.5 - cx
        dy = y + 0.5 - cy
        dz = z + 0.5 - cz
        dx * dx + dy * dy + dz * dz < r2 && return true
    end
    return false
end

function clear_tih2_outside_metal!(tih2, flags)
    @inbounds for n in eachindex(tih2)
        su = flags[n] & TYPE_SU
        if su != TYPE_F && su != TYPE_I
            tih2[n] = 0
        end
    end
    return nothing
end

# One nucleus per call. Nothing is punched when no cell qualifies.
# `tih2_0` is the loading before any release. The hydrogen gate is `c`.
function nucleate_from_tih2!(model, tih2_0, T, fs, c, flags, tags, centers,
                             Tl, H, Nx, Ny, Nz)
    length(centers) >= N_NUC_MAX && return Int32[]
    best = 0.0
    bn = 0
    @inbounds for n in eachindex(tih2_0)
        tih2_0[n] < X_NUC && continue
        c[n] < C_NUC && continue
        fs[n] > FS_LIQUID && continue
        T[n] < Tl && continue
        su = flags[n] & TYPE_SU
        su == TYPE_F || continue
        x, y, z = _xyz(n, Nx, Ny)
        # Stay off the walls and leave metal between the pore and the headspace.
        x > R_NUC + 2 || continue
        y > R_NUC + 2 || continue
        z > R_NUC + 2 || continue
        x < Nx - 1 - (R_NUC + 2) || continue
        y < Ny - 1 - (R_NUC + 2) || continue
        # The free-surface interface sits at z = H. A center in the top
        # liquid cell (z = H-1) would punch into that gas.
        (z + 0.5) + R_NUC <= Float64(H) || continue
        # +1 keeps a cell that has just reached the liquidus in the running.
        score = Float64(tih2_0[n]) * Float64(c[n]) * (Float64(T[n] - Tl) + 1)
        score > best || continue
        best = score
        bn = n
    end
    bn == 0 && return Int32[]
    x, y, z = _xyz(bn, Nx, Ny)
    cx, cy, cz = x + 0.5, y + 0.5, z + 0.5
    near_pore(tags, cx, cy, cz, Nx, Ny, RMIN_NUC) && return Int32[]
    for (px, py, pz) in centers
        dx = px - cx
        dy = py - cy
        dz = pz - cz
        dx * dx + dy * dy + dz * dz < RMIN_NUC * RMIN_NUC && return Int32[]
    end
    ids = try
        spawn_bubbles!(model, [(cx, cy, cz)], [R_NUC])
    catch err
        err isa ArgumentError || rethrow()
        return Int32[]
    end
    isempty(ids) || push!(centers, (cx, cy, cz))
    return ids
end

function metal_means(domain, units)
    flags = Array(domain.flags.data)
    fs = Array(domain.fs.data)
    T = Array(domain.T.data)
    sfs = 0.0
    sT = 0.0
    n = 0
    tmin = Inf
    tmax = -Inf
    twall = Inf
    for i in eachindex(flags)
        if (flags[i] & TYPE_T) != 0x00
            twall = min(twall, Float64(T[i]))
        end
        su = flags[i] & TYPE_SU
        su == TYPE_F || su == TYPE_I || continue
        sfs += Float64(fs[i])
        sT += Float64(T[i])
        n += 1
        tmin = min(tmin, Float64(T[i]))
        tmax = max(tmax, Float64(T[i]))
    end
    n == 0 && return 0.0, 0.0, 0.0, 0.0, 0.0
    return sfs / n, si_T(units, sT / n), si_T(units, tmin), si_T(units, tmax), si_T(units, twall)
end

function max_speed(domain)
    flags = Array(domain.flags.data)
    u = Array(domain.u.data)
    m = 0.0
    for n in 1:size(u, 1)
        su = flags[n] & TYPE_SU
        su == TYPE_F || su == TYPE_I || continue
        s = hypot(Float64(u[n, 1]), Float64(u[n, 2]), Float64(u[n, 3]))
        s > m && (m = s)
    end
    return m
end

function dissolved_mean(domain)
    flags = Array(domain.flags.data)
    c = Array(domain.c.data)
    s = 0.0
    n = 0
    for i in eachindex(flags)
        (flags[i] & TYPE_SU) == TYPE_F || continue
        s += Float64(c[i])
        n += 1
    end
    n == 0 && return 0.0
    return s / n
end

model = Model(Nx, Ny, Nz, units;
              ν=si_ν, α=2 * (si_ν / Pr_heat), gz=-si_g, σ=si_σ, β=0,
              latent=si_L, Ts=si_Ts, Tl=si_Tl, K0=si_K0,
              T_avg=Float32(lbm_T(units, si_Tinit)),
              n_hydro=1)
domain = model.domains[1]
T_init = Float32(lbm_T(units, si_Tinit))
T_dec = Float32(lbm_T(units, si_Tdec))
Tl = Float32(lbm_T(units, si_Tl))
N = Nx * Ny * Nz
flags0 = compact_flags(Nx, Ny, Nz, H)
tih2 = paint_tih2(flags0; seed=seed)
const tih2_0 = copy(tih2)
frac = tih2_mean(tih2, flags0)
@assert abs(frac - TIH2_FRACTION) < 0.03 "TiH2 fraction $frac"
const TIH2_SUM0 = sum(tih2)
Th = fill(T_init, N)
fsh = ones(Float32, N)
copyto!(domain.flags.data, flags0)
copyto!(domain.T.data, Th)
copyto!(domain.fs.data, fsh)
copyto!(domain.mp.data, tih2)
set_foam!(model; D=D, k_H=k_H, k_Π=k_Π, q=0, V_m=V_m, γ_b=1, c0=0, ρ_liquid=1)
initialize!(model)
@assert bubble_count(model) == 0

model.laser = Laser(units;
    P=si_P, w=si_w,
    x=(Nx + 1) / 2, y=(Ny + 1) / 2, z=H + 4,
    dir=(0, 0, -1),
    nrays=11, skin=6, every=1, enabled=false)
qfac = Float64(LatticeBoltzmann.laser_qfac(units))

μ_si = ustrip(u"kg/m^3", si_ρ) * ν_si
@assert abs(Float64(domain.α) / 2 - α_ce) < 1e-4
@printf("316L/TiH2  dx=%.3f mm  dt=%.3e s  μ=%.2f Pa·s  TiH2=%.3f\n",
        1000 * m, units.s, μ_si, frac)
@printf("lattice  ν=%.3f  σ=%.4f  g=%.3e  α=%.4f  Λ=%.3f  Ts=%.3f  Tl=%.3f  Tdec=%.3f\n",
        ν_l, σ_l, g_l, α_ce, Float64(domain.Λ), Float64(domain.Ts), Float64(domain.Tl), T_dec)
@printf("laser  P=%.0f W  w=%.2f cells  on at t=%d  qfac=%.3e  grid %d×%d×%d  H=%d  nsteps=%d\n",
        ustrip(u"W", si_P), Float64(model.laser.w), LASER_ON, qfac, Nx, Ny, Nz, H, nsteps)
@printf("→  %s/lbm.pvd   (mp = TiH2 volume fraction)\n", dir)
flush(stdout)

function molten_count(domain, tih2, Tl)
    flags = Array(domain.flags.data)
    T = Array(domain.T.data)
    fs = Array(domain.fs.data)
    c = Array(domain.c.data)
    n = 0
    nrich = 0
    cmax = 0.0
    for i in eachindex(flags)
        (flags[i] & TYPE_SU) == TYPE_F || continue
        T[i] >= Tl || continue
        fs[i] <= FS_LIQUID || continue
        n += 1
        cmax = max(cmax, Float64(c[i]))
        tih2[i] >= X_NUC && (nrich += 1)
    end
    return n, nrich, cmax
end

function save_frame!(model, domain, units, tih2, t)
    copyto!(domain.mp.data, tih2)
    export!(model; dir, fields, sync=true)
    ids = bubble_ids(model)
    fs̄, TK, Tmin, Tmax, Twall = metal_means(domain, units)
    nmel, nrich, cmax = molten_count(domain, tih2_0, Tl)
    V = 0.0
    for id in ids
        V += bubble_volume(model, id)
    end
    @printf("t=%d  n=%d  V=%.1f  molten=%d  rich=%d  cmax=%.4f  hydride=%.0f%%  c=%.4f  fs=%.3f  T=%.0fK (%.0f–%.0f, wall %.0f)  |u|=%.3f  laser=%s\n",
            t, length(ids), V, nmel, nrich, cmax, 100 * sum(tih2) / TIH2_SUM0,
            dissolved_mean(domain), fs̄, TK, Tmin, Tmax, Twall,
            max_speed(domain), model.laser.enabled ? "on" : "off")
    flush(stdout)
    return ids
end

function main()
    rm(dir; recursive=true, force=true)
    centers = NTuple{3,Float64}[]
    save_frame!(model, domain, units, tih2, 0)
    melted = false
    for t in 1:nsteps
        model.laser.enabled = t >= LASER_ON
        LatticeBoltzmann.step!(model)
        flags = Array(domain.flags.data)
        T = Array(domain.T.data)
        fs = Array(domain.fs.data)
        c = Array(domain.c.data)
        δ = release_hydrogen!(tih2, T, flags, T_dec, Tl, K_REL, C_YIELD)
        add_dissolved!(model, δ)
        @inbounds for n in eachindex(c)
            c[n] += δ[n]
        end
        tags = Array(domain.tag.data)
        ids_new = nucleate_from_tih2!(model, tih2_0, T, fs, c, flags, tags, centers,
                                      Tl, H, Nx, Ny, Nz)
        if !isempty(ids_new)
            flags = Array(domain.flags.data)
            clear_tih2_outside_metal!(tih2, flags)
            @printf("nucleated id=%s at t=%d  (%.2f, %.2f, %.2f)\n",
                    join(string.(ids_new), ","), t, centers[end]...)
            flush(stdout)
        end
        umax = max_speed(domain)
        if umax > 0.4
            save_frame!(model, domain, units, tih2, t)
            error("max |u|=$umax at t=$t exceeds 0.4; lattice foam is unstable")
        end
        _, _, _, Tmax, _ = metal_means(domain, units)
        Tmax >= ustrip(u"K", si_Tl) && (melted = true)
        if t % every == 0 || t == nsteps || !isempty(ids_new)
            save_frame!(model, domain, units, tih2, t)
            # The first steps of a cold solid at this ω_T ring a few tens of
            # kelvin and then settle. The wall check starts once that is over.
            if t >= 200
                _, _, Tmin, _, Twall = metal_means(domain, units)
                if Tmin < Twall - 30
                    error("metal min $(round(Int, Tmin)) K is below the wall $(round(Int, Twall)) K at t=$t")
                end
            end
        end
    end
    if nsteps >= 2500 && bubble_count(model) == 0
        error(melted ? "the spot melted and no TiH2 nucleus formed" :
                       "the spot never reached the liquidus")
    end
    println("wrote $dir/lbm.pvd")
    flush(stdout)
    return bubble_count(model)
end

main()
