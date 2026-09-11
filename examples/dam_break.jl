using LatticeBoltzmann
using Printf
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert SURFACE && VOLUME_FORCE && UPDATE_FIELDS
start_run_log!("output_dam_break")

Nx, Ny, Nz = 64, 64, 64
si_L = 0.1u"m"                  # tank size
si_H = (2 * Nz ÷ 3) / Nz * si_L # dam height ~ 2/3 box height
si_g = 9.81u"m/s^2"
si_u = sqrt(ustrip(u"m/s^2", si_g) * ustrip(u"m", si_H)) * u"m/s"

Ma = 0.05 # D3Q19 Ma is usually safe below 0.05
cs = 1 / sqrt(3)
lbm_u = Ma * cs

# Glycerol at 20 °C against air. Water (ν=1e-6 m²/s, σ=0.072 N/m) is Re~5e4
# at this tank size and is not resolvable on 64³; glycerol gives lattice τ≈0.58.
si_ρ = 1260u"kg/m^3"       # density
ν    = 1.12e-3u"m^2/s"     # kinematic viscosity, μ/ρ ≈ 1.41 Pa·s / 1260 kg/m³
σ    = 0.0634u"N/m"        # surface tension, glycerol–air

units = Units(si_L, si_u, si_ρ; x=Nx, u=lbm_u, ρ=1, T=Float32)
@info "glycerol 20°C" ν σ si_ρ τ=(3 * lbm_ν(units, ν) + 0.5)

model = Model(Nx, Ny, Nz, units;
              ν = ν,                 # kinematic viscosity
              σ = σ,                 # surface tension
              gz = -si_g,            # acceleration due to earth gravity field
              backend=CUDABackend())


host = zeros(UInt8, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
        host[n] = TYPE_S
    elseif x <= Nx ÷ 2 && z <= 2 * Nz ÷ 3   # dam: left half, 2/3 height
        host[n] = TYPE_F
    end
end
copyto!(model.domains[1].flags.data, host)

d = model.domains[1]
LatticeBoltzmann.initialize!(model)
export!(model; dir="output")

nsteps = 2000
every  = 50
nchunks = nsteps ÷ every
Ncell = Int(model.Nx) * Int(model.Ny) * Int(model.Nz)
mlups_ema = NaN
α = 0.2
prog = Progress(nchunks; dt=0.2, desc="dam break ", showspeed=true)
for i in 1:nchunks
    t0 = time_ns()
    with_logger(NullLogger()) do
        run!(model, every)
    end
    dt = (time_ns() - t0) * 1e-9
    mlups = Ncell * every / dt / 1e6
    global mlups_ema = isfinite(mlups_ema) ? α * mlups + (1 - α) * mlups_ema : mlups
    export!(model; dir="output_dam_break")
    next!(prog; showvalues = [
        (:t, Int(d.t)),
        (:t_si, si_t(model.units, Int(d.t))),
        (:MLUPS, round(mlups_ema; digits=1)),
    ])
end
finish!(prog)