# Thermocapillary (Marangoni) convection in an open rectangular cavity.
# Sen & Davis (JFM 1982) / Zebib et al. (Phys. Fluids 1985): hot and cold
# side walls, no-slip bottom, adiabatic free surface. σ = σ0 + σT (T-Tσ).
# Metals: σT < 0, so the surface is pulled toward the cold wall.
#
# Low-Ma Stokes film (closed return flow):
#   u_s ≈ |σT| ΔT H / (4 μ L)
#
#   SURFACE = true, TEMPERATURE = true
using LatticeBoltzmann
using Printf
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE
start_run_log!("output_thermocapillary")

Nx, Ny, Nz = 64, 8, 32
xH, xC = 2, Nx - 1
Hfill = Nz - 4
Lx = Float32(xC - xH)
H  = Float32(Hfill - 1)

ν     = 0.1f0
α     = 0.2f0                 # FluidX3D α; k = α/2 = 0.1, Pr = ν/k = 1
T_hot = 1.5f0
T_cold = 0.5f0
T_avg = 1.0f0
ΔT    = T_hot - T_cold
σ0    = 0.02f0
σT    = -0.02f0               # dσ/dT < 0
k     = 0.1f0                 # thermal_k for α=0.2
Ma    = abs(σT) * ΔT * Lx / (ν * k)
u_est = abs(σT) * ΔT * H / (4 * ν * Lx)

si_L = 0.04u"m"
si_ρ = 900u"kg/m^3"
si_u = 0.01u"m/s"
lbm_u = 0.05
units = Units(si_L, si_u, si_ρ; x=Nx, u=lbm_u, ρ=1, T=Float32, K=300, cp=1500u"J/kg/K")

model = Model(Nx, Ny, Nz, ν;
              α = α, β = 0.0f0, fz = 0.0f0,
              σ = σ0, σT = σT, Tσ = T_avg,
              backend = CUDABackend())
model.units = units
@info "thermocapillary cavity" Ma Pr=(ν/k) σ0 σT Lx H u_est

host = zeros(UInt8, Nx * Ny * Nz)
Th   = fill(T_avg, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if z == 1 || z == Nz || x == 1 || x == Nx
        host[n] = TYPE_S
    elseif z <= Hfill
        if x == xH
            host[n] = TYPE_T
            Th[n] = T_hot
        elseif x == xC
            host[n] = TYPE_T
            Th[n] = T_cold
        else
            host[n] = TYPE_F
            Th[n] = T_hot + (T_cold - T_hot) * Float32(x - xH) / Lx
        end
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].T.data, Th)

d = model.domains[1]
LatticeBoltzmann.initialize!(model)
export!(model; dir="output_thermocapillary")

nsteps = 15000
every  = 250
nchunks = nsteps ÷ every
Ncell = Nx * Ny * Nz
mlups_ema = NaN
αema = 0.2
prog = Progress(nchunks; dt=0.2, desc="thermocapillary ", showspeed=true)
for i in 1:nchunks
    t0 = time_ns()
    with_logger(NullLogger()) do
        run!(model, every)
    end
    dt = (time_ns() - t0) * 1e-9
    mlups = Ncell * every / dt / 1e6
    global mlups_ema = isfinite(mlups_ema) ? αema * mlups + (1 - αema) * mlups_ema : mlups
    export!(model; dir="output_thermocapillary")
    next!(prog; showvalues = [
        (:t, Int(d.t)),
        (:t_si, si_t(model.units, Int(d.t))),
        (:MLUPS, round(mlups_ema; digits=1)),
    ])
end
finish!(prog)

LatticeBoltzmann.moments!(model)
u = Array(d.u.data)
flags = Array(d.flags.data)
ux_s = Float32[]
ux_top = Float32[]
ux_b = Float32[]
zB = 2 + (Hfill - 2) ÷ 3
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    su = flags[n] & TYPE_SU
    if su == TYPE_I && x > xH + 4 && x < xC - 4
        push!(ux_s, u[n, 1])
    elseif su == TYPE_F && z == Hfill && x > xH + 4 && x < xC - 4
        push!(ux_top, u[n, 1])
    elseif su == TYPE_F && z == zB && x > xH + 4 && x < xC - 4
        push!(ux_b, u[n, 1])
    end
end
us = sum(ux_s) / length(ux_s)
ut = sum(ux_top) / length(ux_top)
ub = sum(ux_b) / length(ux_b)
@info "thermocapillary vs Stokes film" Ma u_interface=us u_liquid_top=ut u_return=ub u_est ratio=(ut / u_est)
