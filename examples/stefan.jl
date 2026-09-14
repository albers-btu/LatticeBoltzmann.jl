# One-phase Stefan melting (Neumann, 1860s). Semi-infinite solid at Tm,
# x=0 held at Tb > Tm. Interface X(t) = 2 λ √(α t) with
#   λ exp(λ²) erf(λ) = Ste / √π,   Ste = cp (Tb-Tm) / L.
#
# Edit the SI block. Lattice T = T_SI / Tm so Tm → 1 (VTK T is Kelvin).
# Superheat (Tb-Tm)/Tm should not be ≪ 0.05 or the D3Q7 contrast is weak.
# SURFACE + TEMPERATURE: box is full TYPE_F (no free surface).
using LatticeBoltzmann
using Printf
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert TEMPERATURE
start_run_log!("output_stefan")

@inline function erf_as(x::Float32)
    t = 1.0f0 / (1.0f0 + 0.3275911f0 * abs(x))
    τ = t * (0.254829592f0 + t * (-0.284496736f0 + t * (1.421413741f0 +
        t * (-1.453152027f0 + t * 1.061405429f0))))
    y = 1.0f0 - τ * exp(-x * x)
    return ifelse(x >= 0, y, -y)
end

function stefan_lambda(Ste::Float32)
    rhs = Ste / sqrt(Float32(π))
    lo, hi = 0.01f0, 2.0f0
    λ = 0.3f0
    for _ in 1:50
        λ = 0.5f0 * (lo + hi)
        f = λ * exp(λ^2) * erf_as(λ)
        f > rhs ? (hi = λ) : (lo = λ)
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
xH = 2                                  # Dirichlet hot wall (fluid layer)
xFar = Nx - 1
L = xFar - xH                           # conducting cells (hot wall → far wall)

si_L     = 0.05u"m"                     # bar length
si_ρ     = 2700u"kg/m^3"
si_cp    = 900u"J/kg/K"
si_k     = 100.0u"W/m/K"                # conductivity (liquid metal ~ tens–hundreds)
si_Tm    = 933.0u"K"                    # melting temperature
si_Tb    = 1021.0u"K"                   # hot wall; Ste = cp(Tb-Tm)/L ≈ 0.2
si_Ts    = si_Tm                        # solidus (isothermal: Ts = Tl = Tm)
si_Tl    = si_Tm                        # liquidus
si_Lheat = 3.97e5u"J/kg"                # latent heat
si_K0    = 1.0e-10u"m^2"                # Kozeny–Carman K0 (mush permeability scale)
si_t_end = 8.0u"s"                      # simulated time

α_si = si_k / (si_ρ * si_cp)            # thermal diffusivity
si_ν = ustrip(u"m^2/s", α_si) * u"m^2/s"  # dummy Pr = 1; solid is Darcy-frozen
lbm_α_true = 0.1                        # target lattice diffusivity (stable)
m = ustrip(u"m", si_L) / L
s = lbm_α_true * m^2 / ustrip(u"m^2/s", α_si)
lbm_u = 0.05
si_u = (lbm_u * m / s) * u"m/s"

@assert si_Tb > si_Tm
units = Units(si_L, si_u, si_ρ; x=L, u=lbm_u, ρ=1, T=Float32,
              K=ustrip(u"K", si_Tm), cp=si_cp)

# Model `α` is twice CE diffusivity: thermal_k = α/2.
model = Model(Nx, Ny, Nz, units;
              ν = si_ν,
              α = 2 * α_si,
              β = 0.0f0, gz = 0.0f0, σ = 0,
              latent = si_Lheat,
              Ts = si_Ts, Tl = si_Tl, K0 = si_K0,
              T_avg = Float32(lbm_T(units, si_Tm)),
              backend = CUDABackend())

Tm    = Float32(lbm_T(units, si_Tm))
Tb    = Float32(lbm_T(units, si_Tb))
Ste   = Float32(ustrip(u"J/kg/K", si_cp) * ustrip(u"K", si_Tb - si_Tm) /
                ustrip(u"J/kg", si_Lheat))
k     = thermal_k(model.domains[1])
λ     = stefan_lambda(Ste)
α_m2s = ustrip(u"m^2/s", α_si)
nsteps = max(2, round(Int, ustrip(u"s", si_t_end) / Float64(units.s)))
every  = max(50, nsteps ÷ 40)
ΔT_lat = Tb - Tm
if ΔT_lat < 0.03f0
    @warn "lattice ΔT=$(ΔT_lat) is tiny; increase (Tb-Tm)/Tm or D3Q7 contrast is weak" ΔT_lat Tm Tb
end
@info "Stefan melting" si_Tm si_Tb si_Lheat si_k si_t_end Ste λ Λ=model.domains[1].Λ Tm Tb k nsteps

host = zeros(UInt8, Nx * Ny * Nz)
Th   = fill(Tm, Nx * Ny * Nz)
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
export!(model; dir="output_stefan")

nchunks = nsteps ÷ every
Ncell = Nx * Ny * Nz
mlups_ema = NaN
αema = 0.2
prog = Progress(nchunks; dt=0.2, desc="Stefan ", showspeed=true)
for i in 1:nchunks
    t0 = time_ns()
    with_logger(NullLogger()) do
        run!(model, every)
    end
    dt = (time_ns() - t0) * 1e-9
    mlups = Ncell * every / dt / 1e6
    global mlups_ema = isfinite(mlups_ema) ? αema * mlups + (1 - αema) * mlups_ema : mlups
    export!(model; dir="output_stefan")
    t_lat = Int(d.t)
    t_si  = si_t(model.units, t_lat)
    Xan_mm = 1e3 * 2 * λ * sqrt(α_m2s * t_si)
    next!(prog; showvalues = [
        (:t, t_lat),
        (:t_si, round(t_si; digits=3)),
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
@info "Stefan vs Neumann" Ste λ Xnum_mm=(1e3 * Xnum_m) Xan_mm=(1e3 * Xan_m) ratio=(Xnum_m / Xan_m) t_end
