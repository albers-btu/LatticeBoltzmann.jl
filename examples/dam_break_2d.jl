# 2D dam break. Walls on x and y only. Height is y, so gravity is gy.
# σT = 0 keeps Marangoni off. Nz = 1 is not a wall.
using LatticeBoltzmann
using Printf
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert SURFACE && VOLUME_FORCE && UPDATE_FIELDS
start_run_log!("output_dam_break_2d")

# In-plane tank. Not the 256³ allocation: Nz = 1.
# 128² is coarser than that tank, so water ν can sit under the KBC
# floor (ω≤1.999) at Ma=0.05. Same water properties either way.
Nx, Ny, Nz = 128, 128, 1
si_L = 0.1u"m"                  # tank size
si_H = (2 * Ny ÷ 3) / Ny * si_L # dam height ~ 2/3 box height
si_g = 9.81u"m/s^2"
si_u = sqrt(ustrip(u"m/s^2", si_g) * ustrip(u"m", si_H)) * u"m/s"

Ma = 0.05 # lattice Ma is usually safe below 0.05
cs = 1 / sqrt(3)
lbm_u = Ma * cs

# Water at 20 °C.
si_ρ = 998u"kg/m^3"
ν    = 1.004e-6u"m^2/s"
σ    = 0.0728u"N/m"

units = Units(si_L, si_u, si_ρ; x=Nx, u=lbm_u, ρ=1, T=Float32)
ν_lat = lbm_ν(units, ν)
ν_floor = (1 / 1.999 - 0.5) / 3
σ_lat = lbm_σ(units, σ)
if ν_lat < ν_floor
    @warn "water ν is below the KBC floor on this grid; effective ν is higher" ν_lat ν_floor ν_eff_over_real=(ν_floor / ν_lat)
end
@info "water 20°C" ν σ si_ρ ν_lat σ_lat τ=(3 * max(ν_lat, ν_floor) + 0.5)

model = Model(Nx, Ny, Nz, units;
              ν = ν,                 # kinematic viscosity
              σ = σ,                 # surface tension
              σT = 0,                # Marangoni off
              gy = -si_g,            # in-plane gravity
              backend=CUDABackend())


host = zeros(UInt8, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if x == 1 || x == Nx || y == 1 || y == Ny
        host[n] = TYPE_S
    elseif x <= Nx ÷ 2 && y <= 2 * Ny ÷ 3   # dam: left half, 2/3 height
        host[n] = TYPE_F
    end
end
copyto!(model.domains[1].flags.data, host)

d = model.domains[1]
LatticeBoltzmann.initialize!(model)
export!(model; dir="output_dam_break_2d", fields=(:rho, :p, :u, :phi, :flags))

si_t_end = 0.112u"s"   # same physical duration as the 3D dam break
nsteps = max(1, round(Int, ustrip(u"s", si_t_end) / Float64(units.s)))
every  = max(1, round(Int, nsteps / 40))
nchunks = nsteps ÷ every
Ncell = Int(model.Nx) * Int(model.Ny) * Int(model.Nz)
mlups_ema = NaN
α = 0.2
prog = Progress(nchunks; dt=0.2, desc="dam break 2d ", showspeed=true)
for i in 1:nchunks
    t0 = time_ns()
    with_logger(NullLogger()) do
        run!(model, every)
    end
    dt = (time_ns() - t0) * 1e-9
    mlups = Ncell * every / dt / 1e6
    global mlups_ema = isfinite(mlups_ema) ? α * mlups + (1 - α) * mlups_ema : mlups
    export!(model; dir="output_dam_break_2d", fields=(:rho, :p, :u, :phi, :flags))
    next!(prog; showvalues = [
        (:t, Int(d.t)),
        (:t_si, si_t(model.units, Int(d.t))),
        (:MLUPS, round(mlups_ema; digits=1)),
    ])
end
finish!(prog)
