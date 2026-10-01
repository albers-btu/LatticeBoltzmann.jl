# Aluminum foam in an open mold. Requires FOAM, SURFACE, and TEMPERATURE.
# n_hydro stays 1.
#
# The melt is Al–Si (ρ, cp, k, σ, latent heat, solidus, liquidus). It is
# poured above the liquidus into a mold below the solidus and cools by
# conduction into the walls. The free surface and the pore gas are
# insulating: gas cells do not carry heat. Pure liquid aluminum is
# about 10⁻³ Pa·s. On this grid that viscosity cannot carry both
# gravity and surface tension, so the melt is the stabilized foam
# viscosity. With μ ≈ 0.8 Pa·s and the conductivity of liquid aluminum
# the Prandtl number is 10, and a centimetre-wide casting freezes
# before the cells can inflate. The stabilized melt used here is
# μ ≈ 3.3 Pa·s (more oxide and particles), Pr = 40, so the core stays
# liquid for the expansion. Lattice ν stays 0.2. The cell size still
# puts g = 9.81 m/s² at |g_lat| = 2×10⁻⁵.
#
# Equilibrium hydrogen in aluminum is only a few percent gas at 1 bar.
# The blowing agent is a lattice inventory, not that solubility:
# final gas volume ≈ V_m c0 × (liquid volume) once the solute has
# entered the pores. V_m c0 = 1.5, so a full conversion is about
# 60% gas. The free surface rises by about 1.5 pour-heights and the
# films between the pores are left at about one cell.
# d(R²)/dt = 2 c0 V_m D = 0.030 while the bath is still rich.
# Nuclei (R0 = 4) are rmin = 12 apart, so the initial film is
# 4 cells and disjoining (range 4) is off until a film thins.
# They sit R0+4 cells off the wall, under an 8-cell liquid cap.
# The free surface is not a Henry sink: tag −1 reconstructs at the
# concentration of the liquid face neighbors. A one-cell gas bridge
# into that headspace is turned back into liquid, so a pore that
# touches the atmosphere loses the finger and keeps its gas.
# Henry's anti-bounce-back drop is put back on the liquid. The pore
# is credited only the liquid-side flux, and that same amount is
# taken off the liquid, so the bath falls as the cells inflate.
# The mold is preheated to 800 K,
# below the solidus. A bubble whose whole shell has liquid fraction
# below 10⁻³ keeps its id, its gas mass, and its volume.
#
# Open output_foam_free/lbm.pvd. Colour by tag (0 liquid, −1 atmosphere)
# or by fs (0 liquid, 1 solid). Contour phi = 0.5.
# ParaView 6: do not drag an isosurface on a flat array.
#
# A frame is written every 250 steps. axis is how many bubbles have their
# farthest interface cell within 15° of ±x, ±y, or ±z.

using LatticeBoltzmann
using Printf
using Unitful

@assert FOAM && SURFACE && TEMPERATURE

# --- Al–Si melt, cold mold ---
si_ρ     = 2400u"kg/m^3"
si_cp    = 1100u"J/kg/K"
si_k     = 90u"W/m/K"
si_Ts    = 850u"K"
si_Tl    = 890u"K"
si_Tinit = 1020u"K"
# Preheated mold. Still below the solidus (850 K), so the casting
# freezes, but the front is slow enough for the cells to inflate.
si_Tmold = 800u"K"
si_L     = 3.97e5u"J/kg"
si_σ     = 0.87u"N/m"
si_g     = 9.81u"m/s^2"
si_K0    = 1.0e-10u"m^2"

α_si = si_k / (si_ρ * si_cp)
# ν_lat / α_CE = Pr. Pr = 40 with ν_lat = 0.2 ⇒ ν_si = 40 α_si,
# μ = ρ ν_si ≈ 3.3 Pa·s. Lattice diffusivity drops by the same factor,
# so the freeze front crosses the mold several times more slowly.
ν_lat = 0.2
Pr = 40
si_ν = Pr * α_si
ν_si = ustrip(u"m^2/s", si_ν)
g_si = ustrip(u"m/s^2", si_g)
g_lat_target = 2.0e-5
# g_lat = g s²/m and s = ν_lat m²/ν_si, so g_lat ∝ m³.
m = cbrt(g_lat_target * ν_si^2 / (g_si * ν_lat^2))
s = ν_lat * m^2 / ν_si
lbm_u = 0.05
si_u = (lbm_u * m / s) * u"m/s"
units = Units(m * u"m", si_u, si_ρ; x=1, u=lbm_u, ρ=1, T=Float32,
              K=ustrip(u"K", si_Tl), cp=si_cp)

ν_l = Float64(lbm_ν(units, si_ν))
σ_l = Float64(lbm_σ(units, si_σ))
g_l = Float64(lbm_g(units, -si_g))
# Model α is twice the CE diffusivity. α_CE = ν_lat / Pr.
α_ce = ν_lat / Pr
@assert 0.19 < ν_l < 0.21
@assert 0.001 < σ_l < 0.06
@assert -3.0e-5 < g_l < -1.2e-5
@assert 0.004 < α_ce < 0.006

# Half-width of the melt. With the mold 50 K under the solidus the
# solid skin is about 1.5 sqrt(α t) cells thick, so the center is
# solid once that reaches the half-width. The pores sit R0+4 cells
# inboard of the wall and take their gas on the diffusion time of
# the film, which is shorter than the time for the skin to arrive.
# 56 across is one step up from 48. The cell size is unchanged, so
# |g_lat| stays 2×10⁻⁵. R0 grows with the grid so a pore is 8 cells
# across. The film stays 4 cells (disjoining range).
Nx = 56
Ny = 56
Nz = 131
H = 56
Xfreeze = (min(Nx, Ny) - 2) / 2
n_cool = ceil(Int, 0.45 * Xfreeze^2 / α_ce)
n_hold = 400
nsteps = n_cool + n_hold
R0 = 4.0
# Shells must not overlap: rmin >= 2R+1. rmin = 2 R0 + 4 leaves a
# 4-cell film, outside the disjoining range, so Π starts at zero.
rmin = 12.0
margin = R0 + 4
cap = 8
seed = 1
# Blowing agent. V_m * c0 = 1.5 → about 60% gas if the solute enters
# the pores. d(R²)/dt = 2*c0*V_m*D = 0.030. A rate near 4 (Fig. 15
# with Δc = 0.5 and V_m = 150) blew the lattice Mach number.
k_H = 1e-5
D = 0.01
V_m = 30.0
c0 = 0.05
k_Π = 0.02
dir = "output_foam_free"
fields = (:phi, :c, :tag, :rho, :u, :flags, :T, :fs)

function mold_flags(Nx, Ny, Nz, H)
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

function bubble_radius(model, id)
    return (3 * bubble_volume(model, id) / (4π))^(1 / 3)
end

function mean_radius(model, ids)
    isempty(ids) && return 0.0
    s = 0.0
    for id in ids
        s += bubble_radius(model, id)
    end
    return s / length(ids)
end

function total_volume(model, ids)
    s = 0.0
    for id in ids
        s += bubble_volume(model, id)
    end
    return s
end

# Solid plus liquid metal, from the fill fraction. Walls and pore gas
# are left out, so Vg / (Vg + Vm) is the porosity of the expanded charge.
function metal_volume(domain)
    flags = Array(domain.flags.data)
    ϕ = Array(domain.ϕ.data)
    s = 0.0
    for i in eachindex(flags)
        su = flags[i] & TYPE_SU
        if su == TYPE_F
            s += 1
        elseif su == TYPE_I
            p = Float64(ϕ[i])
            s += p < 0 ? 0.0 : (p > 1 ? 1.0 : p)
        end
    end
    return s
end

function porosity(model, domain, ids)
    Vg = total_volume(model, ids)
    Vm = metal_volume(domain)
    return Vg / max(Vg + Vm, 1e-6)
end

# Highest cell that is still liquid or interface. The lid is Nz − 1.
function liquid_level(domain)
    Nx = Int(domain.Nx)
    Ny = Int(domain.Ny)
    Nz = Int(domain.Nz)
    flags = Array(domain.flags.data)
    top = 0
    for z in 0:(Nz - 1), y in 0:(Ny - 1), x in 0:(Nx - 1)
        su = flags[1 + x + Nx * y + Nx * Ny * z] & TYPE_SU
        if su == TYPE_F || su == TYPE_I
            top = max(top, z)
        end
    end
    return top
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

# Liquid and interface only. Gas cells skip the collide, so a velocity
# left on them when the interface advanced is not a lattice speed.
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

function mean_ratio(model, ids)
    isempty(ids) && return 0.0
    s = 0.0
    for id in ids
        s += bubble_ratio(model, id)
    end
    return s / length(ids)
end

# Dissolved blowing agent left in the liquid. It starts at c0 and
# should fall as the pores take it up.
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

# Solid fraction in the inner half of the pour. This is the film
# region; the wall skin is outside it.
function core_solid(domain, H)
    Nx = Int(domain.Nx)
    Ny = Int(domain.Ny)
    Nz = Int(domain.Nz)
    flags = Array(domain.flags.data)
    fs = Array(domain.fs.data)
    x0 = Nx ÷ 4
    x1 = Nx - 1 - x0
    y0 = Ny ÷ 4
    y1 = Ny - 1 - y0
    z1 = min(Nz - 2, H - 2)
    s = 0.0
    n = 0
    for z in 2:z1, y in y0:y1, x in x0:x1
        i = 1 + x + Nx * y + Nx * Ny * z
        su = flags[i] & TYPE_SU
        su == TYPE_F || su == TYPE_I || continue
        s += Float64(fs[i])
        n += 1
    end
    n == 0 && return 0.0
    return s / n
end

function n_frozen(model, ids)
    n = 0
    for id in ids
        bubble_frozen(model, id) && (n += 1)
    end
    return n
end

# Bubbles whose farthest interface cell lies within 15° of a grid axis.
function axis_tips(domain, ids)
    Nx = Int(domain.Nx)
    Ny = Int(domain.Ny)
    Nz = Int(domain.Nz)
    tags = Array(domain.tag.data)
    ϕ = Array(domain.ϕ.data)
    sx = Dict{Int32,Float64}(id => 0.0 for id in ids)
    sy = Dict{Int32,Float64}(id => 0.0 for id in ids)
    sz = Dict{Int32,Float64}(id => 0.0 for id in ids)
    sw = Dict{Int32,Float64}(id => 0.0 for id in ids)
    shell = Dict{Int32,Vector{NTuple{3,Float64}}}(id => NTuple{3,Float64}[] for id in ids)
    for z in 0:(Nz - 1), y in 0:(Ny - 1), x in 0:(Nx - 1)
        n = 1 + x + Nx * y + Nx * Ny * z
        id = tags[n]
        haskey(sw, id) || continue
        w = 1 - ϕ[n]
        w > 1e-4 || continue
        cx = x + 0.5
        cy = y + 0.5
        cz = z + 0.5
        sx[id] += w * cx
        sy[id] += w * cy
        sz[id] += w * cz
        sw[id] += w
        if 0.05 < ϕ[n] < 0.95
            push!(shell[id], (cx, cy, cz))
        end
    end
    naxis = 0
    nused = 0
    for id in ids
        sw[id] == 0 && continue
        isempty(shell[id]) && continue
        gx = sx[id] / sw[id]
        gy = sy[id] / sw[id]
        gz = sz[id] / sw[id]
        best = 0.0
        dir = (0.0, 0.0, 0.0)
        for (cx, cy, cz) in shell[id]
            dx = cx - gx
            dy = cy - gy
            dz = cz - gz
            r2 = dx * dx + dy * dy + dz * dz
            r2 > best || continue
            best = r2
            dir = (dx, dy, dz)
        end
        best == 0 && continue
        r = sqrt(best)
        ax = max(abs(dir[1]), abs(dir[2]), abs(dir[3])) / r
        nused += 1
        acosd(min(ax, 1.0)) < 15 && (naxis += 1)
    end
    return naxis, nused
end

model = Model(Nx, Ny, Nz, units;
              ν=si_ν, α=2 * α_si, gz=-si_g, σ=si_σ, β=0,
              latent=si_L, Ts=si_Ts, Tl=si_Tl, K0=si_K0,
              T_avg=Float32(lbm_T(units, si_Tinit)),
              n_hydro=1)
domain = model.domains[1]
T_init = Float32(lbm_T(units, si_Tinit))
T_mold = Float32(lbm_T(units, si_Tmold))
N = Nx * Ny * Nz
flags = mold_flags(Nx, Ny, Nz, H)
Th = fill(T_init, N)
fsh = zeros(Float32, N)
for i in eachindex(flags)
    if (flags[i] & TYPE_T) != 0x00
        Th[i] = T_mold
        fsh[i] = 1
    end
end
copyto!(domain.flags.data, flags)
copyto!(domain.T.data, Th)
copyto!(domain.fs.data, fsh)

z_hi = H - cap
centers = poisson_disk_centers(Nx, Ny, z_hi, rmin; seed=seed, margin=margin)
@assert length(centers) >= 2 "poisson placed $(length(centers)) nuclei; widen the mold or lower rmin"
nucleate_bubbles!(model, centers, fill(R0, length(centers)))
set_foam!(model; D=D, k_H=k_H, k_Π=k_Π, q=0, V_m=V_m, γ_b=1,
          c0=c0, ρ_liquid=1)
initialize!(model)

every = 250
μ_si = ustrip(u"kg/m^3", si_ρ) * ν_si
@assert abs(Float64(domain.α) / 2 - α_ce) < 1e-4
@printf("Al–Si foam  dx=%.3f mm  dt=%.3e s  μ=%.2f Pa·s\n", 1000 * m, units.s, μ_si)
@printf("lattice  ν=%.3f  σ=%.4f  g=%.3e  α=%.4f  Λ=%.3f  Ts=%.3f  Tl=%.3f  Tmold=%.3f\n",
        ν_l, σ_l, g_l, α_ce, Float64(domain.Λ), Float64(domain.Ts), Float64(domain.Tl), T_mold)
@printf("n0=%d  H=%d  cap=%d  k_Π=%g  c0=%g  V_m=%g  D=%g  nsteps=%d  →  %s/lbm.pvd\n",
        length(centers), H, cap, k_Π, c0, V_m, D, nsteps, dir)
flush(stdout)

function save_frame!(model, domain, units, t)
    export!(model; dir, fields, sync=true)
    ids = bubble_ids(model)
    naxis, nused = axis_tips(domain, ids)
    fs̄, TK, Tmin, Tmax, Twall = metal_means(domain, units)
    umax = max_speed(domain)
    φ = porosity(model, domain, ids)
    @printf("t=%d  n=%d  frozen=%d  meanR=%.3f  V=%.1f  φ=%.2f  level=%d  axis=%d/%d  fs=%.3f  core=%.3f  c=%.4f  ratio=%.2f  T=%.0fK (%.0f–%.0f, wall %.0f)  |u|=%.3f  ids=%s\n",
            t, length(ids), n_frozen(model, ids), mean_radius(model, ids),
            total_volume(model, ids), φ, liquid_level(domain), naxis, nused,
            fs̄, core_solid(domain, H), dissolved_mean(domain), mean_ratio(model, ids),
            TK, Tmin, Tmax, Twall, umax, join(string.(ids), ","))
    flush(stdout)
    return ids, umax
end

function main()
    rm(dir; recursive=true, force=true)
    save_frame!(model, domain, units, 0)
    held = 0
    V_at_freeze = 0.0
    n_at_freeze = 0
    ids_at_freeze = Int32[]
    for t in 1:nsteps
        LatticeBoltzmann.step!(model)
        ids = bubble_ids(model)
        umax = max_speed(domain)
        if umax > 0.4
            save_frame!(model, domain, units, t)
            error("max |u|=$umax at t=$t exceeds 0.4; lattice foam is unstable")
        end
        all_frozen = !isempty(ids) && all(id -> bubble_frozen(model, id), ids)
        if all_frozen
            if held == 0
                V_at_freeze = total_volume(model, ids)
                n_at_freeze = length(ids)
                ids_at_freeze = copy(ids)
            end
            held += 1
        else
            held = 0
        end
        if t % every == 0 || t == nsteps || held == 1 || held == 400
            save_frame!(model, domain, units, t)
            _, _, Tmin, _, Twall = metal_means(domain, units)
            if Tmin < Twall - 30
                error("melt min $(round(Int, Tmin)) K is below the mold $(round(Int, Twall)) K at t=$t")
            end
        end
        if held == 400
            ids_now = bubble_ids(model)
            V_now = total_volume(model, ids_now)
            n_now = length(ids_now)
            same = n_now == n_at_freeze && issetequal(ids_now, ids_at_freeze) &&
                   abs(V_now - V_at_freeze) / max(V_at_freeze, 1e-6) < 0.02
            same || error("frozen foam moved: n $n_at_freeze→$n_now  V $V_at_freeze→$V_now")
            φ = porosity(model, domain, ids_now)
            level = liquid_level(domain)
            # A metal foam, not a few small holes in a solid block.
            # Porosity 0.45 is past the sphere-packing films; the free
            # surface has to have risen with the gas.
            φ >= 0.45 || error("frozen porosity $φ is below 0.45; the charge did not foam")
            level >= H + 8 || error("free surface at $level did not rise above the pour at $H")
            n_now >= 6 || error("only $n_now pores survived")
            @printf("held a frozen foam for 400 steps  n=%d  V=%.1f  φ=%.2f  level=%d\n",
                    n_now, V_now, φ, level)
            flush(stdout)
            break
        end
    end
    if held < 400
        error("stopped at t=$(Int(domain.t)) with frozen hold = $held (wanted 400)")
    end
    println("wrote $dir/lbm.pvd")
    flush(stdout)
    return held
end

main()
