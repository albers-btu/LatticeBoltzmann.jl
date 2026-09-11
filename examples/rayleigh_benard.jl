# Rayleigh–Bénard convection (FluidX3D setup.cpp). D3Q7 temperature + Boussinesq.
# Bottom TYPE_T hot, top TYPE_T cold, outer z faces TYPE_S; x/y periodic.
# Ra = |g| β ΔT H³ / (ν α). Rolls for Ra ≳ 1708 (no-slip).
#
# Lattice T must have O(1) contrast (FluidX3D uses 1.75 / 0.25). Mapping a
# 10 K air gap onto T=1±0.017 makes buoyancy a 2% ripple on full g and
# kills the rolls. SI here is only box size / time for VTK.
#
#   TEMPERATURE  = true
#   TRT          = true
using LatticeBoltzmann
using Printf
using CUDA
using Unitful
using ProgressMeter
using Logging
using Random

@assert TEMPERATURE
start_run_log!("output_rb")

Nx, Ny, Nz = 256, 8, 32
ν     = 0.02f0
α     = 0.02f0                   # Pr = 1
β     = 0.2f0
T_avg = 1.0f0
T_hot = 1.75f0
T_cold = 0.25f0
g     = 0.0005f0
H     = Float32(Nz - 2)
ΔT    = T_hot - T_cold
Ra    = g * β * ΔT * H^3 / (ν * α)
lbm_u = sqrt(g * β * ΔT * H)

si_H = 0.05u"m"
si_L = Nx / (Nz - 2) * si_H
si_ρ = 1.204u"kg/m^3"
si_u = 0.05u"m/s"
units = Units(si_L, si_u, si_ρ; x=Nx, u=lbm_u, ρ=1, T=Float32, K=300)
@info "Rayleigh–Bénard" Ra Pr=(ν/α) ν α β g H lbm_u

model = Model(Nx, Ny, Nz, ν;
              fz = -g, α = α, β = β, T_avg = T_avg,
              backend = CUDABackend())
model.units = units

host = zeros(UInt8, Nx * Ny * Nz)
Th   = fill(T_avg, Nx * Ny * Nz)
uh   = zeros(Float32, Nx * Ny * Nz, 3)
ρh   = ones(Float32, Nx * Ny * Nz)
z0   = 0.5f0 * (Nz + 1)
rng = MersenneTwister(1)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    ρh[n] = 1 + 3 * g * (z0 - z)    # hydrostatic, p = ρ/3
    if z == 1 || z == Nz
        host[n] = TYPE_S
    elseif z == 2
        host[n] = TYPE_T
        Th[n] = T_hot
        uh[n, 1] = 0.015f0 * (rand(rng, Float32) - 0.5f0)
        uh[n, 3] = 0.015f0 * (rand(rng, Float32) - 0.5f0)
    elseif z == Nz - 1
        host[n] = TYPE_T
        Th[n] = T_cold
    else
        uh[n, 1] = 0.015f0 * (rand(rng, Float32) - 0.5f0)
        uh[n, 3] = 0.015f0 * (rand(rng, Float32) - 0.5f0)
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].T.data, Th)
copyto!(model.domains[1].u.data, uh)
copyto!(model.domains[1].ρ.data, ρh)

d = model.domains[1]
LatticeBoltzmann.initialize!(model)
export!(model; dir="output_rb")

nsteps = 30000
every  = 100
nchunks = nsteps ÷ every
Ncell = Nx * Ny * Nz
mlups_ema = NaN
αema = 0.2
prog = Progress(nchunks; dt=0.2, desc="Rayleigh–Bénard ", showspeed=true)
for i in 1:nchunks
    t0 = time_ns()
    with_logger(NullLogger()) do
        run!(model, every)
    end
    dt = (time_ns() - t0) * 1e-9
    mlups = Ncell * every / dt / 1e6
    global mlups_ema = isfinite(mlups_ema) ? αema * mlups + (1 - αema) * mlups_ema : mlups
    export!(model; dir="output_rayleigh_benard")
    next!(prog; showvalues = [
        (:t, Int(d.t)),
        (:t_si, si_t(model.units, Int(d.t))),
        (:MLUPS, round(mlups_ema; digits=1)),
    ])
end
finish!(prog)
