# Two-phase Neumann melting (Carslaw & Jaeger). Semi-infinite solid at Ti < Tm,
# x=0 held at Tb > Tm, k_s ≠ k_l. Interface X(t) = 2 λ √(α_l t) with
#   λ√π = Ste_l e^{-λ²}/erf(λ) - Ste_s √(α_s/α_l) e^{-λ² α_l/α_s}/erfc(λ√(α_l/α_s))
#   Ste_l = cp(Tb-Tm)/L,  Ste_s = cp(Tm-Ti)/L.
#
# Lattice: α(f_s), ν(f_s). Model `α` is twice CE diffusivity (k = α/2).
# Edit the SI block. Closed TYPE_F box (no free surface).
using LatticeBoltzmann
using Printf
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert TEMPERATURE
start_run_log!("output_neumann_twophase")

@inline function erf_as(x::Float32)
    t = 1.0f0 / (1.0f0 + 0.3275911f0 * abs(x))
    τ = t * (0.254829592f0 + t * (-0.284496736f0 + t * (1.421413741f0 +
        t * (-1.453152027f0 + t * 1.061405429f0))))
    y = 1.0f0 - τ * exp(-x * x)
    return ifelse(x >= 0, y, -y)
end
@inline erfc_as(x::Float32) = 1.0f0 - erf_as(x)

function neumann_lambda_two_phase(Ste_l::Float32, Ste_s::Float32, κs_over_κl::Float32)
    r = sqrt(κs_over_κl)
    invr = 1.0f0 / r
    lo, hi = 0.01f0, 2.0f0
    λ = 0.2f0
    for _ in 1:60
        λ = 0.5f0 * (lo + hi)
        t1 = Ste_l * exp(-λ * λ) / erf_as(λ)
        t2 = Ste_s * r * exp(-λ * λ * invr * invr) / erfc_as(λ * invr)
        f = t1 - t2 - λ * sqrt(Float32(π))
        f > 0 ? (lo = λ) : (hi = λ)
    end
    return λ
end

function interface_x(fsA, Nx, Ny, Nz)
    y0, z0 = Ny ÷ 2, Nz ÷ 2
    for x in 3:Nx-1
        n = x + (y0 - 1) * Nx + (z0 - 1) * Nx * Ny
        fsA[n] > 0.5f0 && return Float32(x - 2)
    end
    return Float32(Nx - 3)
end

# --- SI inputs ---
Nx, Ny, Nz = 96, 8, 8
xH = 2
xFar = Nx - 1
L = xFar - xH

si_L     = 0.05u"m"
si_ρ     = 2700u"kg/m^3"
si_cp    = 900u"J/kg/K"
si_k_l   = 80.0u"W/m/K"                 # liquid conductivity
si_k_s   = 160.0u"W/m/K"                # solid (typically higher)
si_Tm    = 933.0u"K"
si_Tb    = 1021.0u"K"                   # hot wall
si_Ti    = 845.0u"K"                    # initial solid
si_Lheat = 3.97e5u"J/kg"
si_K0    = 1.0e-10u"m^2"
si_t_end = 8.0u"s"

α_l_si = si_k_l / (si_ρ * si_cp)
α_s_si = si_k_s / (si_ρ * si_cp)
si_ν_l = ustrip(u"m^2/s", α_l_si) * u"m^2/s"
si_ν_s = 0.5 * si_ν_l
lbm_α_true = 0.1                        # target liquid CE diffusivity
m = ustrip(u"m", si_L) / L
s = lbm_α_true * m^2 / ustrip(u"m^2/s", α_l_si)
lbm_u = 0.05
si_u = (lbm_u * m / s) * u"m/s"

@assert si_Tb > si_Tm > si_Ti
units = Units(si_L, si_u, si_ρ; x=L, u=lbm_u, ρ=1, T=Float32,
              K=ustrip(u"K", si_Tm), cp=si_cp)

model = Model(Nx, Ny, Nz, units;
              ν = si_ν_l,
              α = 2 * α_l_si,
              α_s = 2 * α_s_si,
              α_l = 2 * α_l_si,
              ν_s = si_ν_s,
              ν_l = si_ν_l,
              β = 0.0f0, gz = 0.0f0, σ = 0,
              latent = si_Lheat,
              Ts = si_Tm, Tl = si_Tm, K0 = si_K0,
              T_avg = Float32(lbm_T(units, si_Tm)),
              backend = CUDABackend())

Tm    = Float32(lbm_T(units, si_Tm))
Tb    = Float32(lbm_T(units, si_Tb))
Ti    = Float32(lbm_T(units, si_Ti))
Ste_l = Float32(ustrip(u"J/kg/K", si_cp) * ustrip(u"K", si_Tb - si_Tm) /
                ustrip(u"J/kg", si_Lheat))
Ste_s = Float32(ustrip(u"J/kg/K", si_cp) * ustrip(u"K", si_Tm - si_Ti) /
                ustrip(u"J/kg", si_Lheat))
k_l   = thermal_k_l(model.domains[1])
k_s   = thermal_k_s(model.domains[1])
λ     = neumann_lambda_two_phase(Ste_l, Ste_s, k_s / k_l)
α_m2s = ustrip(u"m^2/s", α_l_si)
nsteps = max(2, round(Int, ustrip(u"s", si_t_end) / Float64(units.s)))
every  = max(50, nsteps ÷ 40)
@info "two-phase Neumann" si_Tm si_Tb si_Ti si_k_l si_k_s Ste_l Ste_s λ k_l k_s nsteps

host = zeros(UInt8, Nx * Ny * Nz)
Th   = fill(Ti, Nx * Ny * Nz)
fsh  = ones(Float32, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
        host[n] = TYPE_S
    elseif x == xH
        host[n] = TYPE_T
        Th[n] = Tb
        fsh[n] = 0
    else
        host[n] = TYPE_F
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].T.data, Th)
copyto!(model.domains[1].fs.data, fsh)

d = model.domains[1]
LatticeBoltzmann.initialize!(model)
export!(model; dir="output_neumann_twophase")

nchunks = nsteps ÷ every
Ncell = Nx * Ny * Nz
mlups_ema = NaN
αema = 0.2
prog = Progress(nchunks; dt=0.2, desc="two-phase Neumann ", showspeed=true)
for i in 1:nchunks
    t0 = time_ns()
    with_logger(NullLogger()) do
        run!(model, every)
    end
    dt = (time_ns() - t0) * 1e-9
    mlups = Ncell * every / dt / 1e6
    global mlups_ema = isfinite(mlups_ema) ? αema * mlups + (1 - αema) * mlups_ema : mlups
    export!(model; dir="output_neumann_twophase")
    t_lat = Int(d.t)
    t_now = si_t(model.units, t_lat)
    Xan_mm = 1e3 * 2 * λ * sqrt(α_m2s * t_now)
    next!(prog; showvalues = [
        (:t, t_lat),
        (:t_si, round(t_now; digits=3)),
        (:X_an_mm, round(Xan_mm; digits=2)),
        (:MLUPS, round(mlups_ema; digits=1)),
    ])
end
finish!(prog)

LatticeBoltzmann.moments!(model)
Xnum_cells = interface_x(Array(d.fs.data), Nx, Ny, Nz)
t_end = si_t(model.units, nsteps)
Xnum_m = Xnum_cells * Float32(units.m)
Xan_m  = Float32(2 * λ * sqrt(α_m2s * t_end))
@info "two-phase vs Neumann" Ste_l Ste_s λ Xnum_mm=(1e3 * Xnum_m) Xan_mm=(1e3 * Xan_m) ratio=(Xnum_m / Xan_m) t_end
