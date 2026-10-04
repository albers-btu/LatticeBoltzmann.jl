# 2D velocity-driven channel. Classic TYPE_E use:
#   x = 1 and x = Nx -> equilibrium BC, prescribed ρ=1, u=(u_in,0,0)
#   y = 1 and y = Ny -> bounce-back walls
#   Nz = 1 is the plane, not a wall (a z face would solidify every cell)
#   EQUILIBRIUM_BOUNDARIES = true
using LatticeBoltzmann
using Printf
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert EQUILIBRIUM_BOUNDARIES
start_run_log!("output_channel_2d")

Nx, Ny, Nz = 48, 24, 1
si_L = 0.048u"m"                    # streamwise length (cubic cells)
si_H = Ny / Nx * si_L               # channel height is the y-span

# Air at 20 °C. Water at this size/speed is Re~2e3 and ω->2 on 48×24.
si_ρ = 1.204u"kg/m^3"
ν    = 1.51e-5u"m^2/s"              # kinematic viscosity of air
si_u = 0.10u"m/s"                   # inlet / outlet speed

Ma = 0.08                           # lattice Mach of si_u
cs = 1 / sqrt(3)
lbm_inlet = Ma * cs

units = Units(si_L, si_u, si_ρ; x=Nx, u=lbm_inlet, ρ=1, T=Float32)
Re = ustrip(u"m/s", si_u) * ustrip(u"m", si_H) / ustrip(u"m^2/s", ν)
@info "air 20°C" ν si_ρ si_u si_L si_H Re τ=(3 * lbm_ν(units, ν) + 0.5)

model = Model(Nx, Ny, Nz, units;
              ν = ν,
              gx = 0, gy = 0, gz = 0,
              backend = CUDABackend())

u_in = Float32(lbm_u(units, si_u))  # lattice inlet speed (= lbm_inlet)
host = zeros(UInt8, Nx * Ny * Nz)
uh   = zeros(Float32, Nx * Ny * Nz, 3)
ρh   = ones(Float32, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if x == 1 || x == Nx
        host[n] = TYPE_E
        uh[n, 1] = u_in
    elseif y == 1 || y == Ny
        host[n] = TYPE_S
    else
        # Flag 0 is gas when SURFACE is on. The duct has to be fluid.
        host[n] = TYPE_F
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].u.data, uh)
copyto!(model.domains[1].ρ.data, ρh)

d = model.domains[1]
LatticeBoltzmann.initialize!(model)
export!(model; dir="output_channel_2d")

nsteps = 2000
every  = 50
nchunks = nsteps ÷ every
Ncell = Int(model.Nx) * Int(model.Ny) * Int(model.Nz)
mlups_ema = NaN
α = 0.2
prog = Progress(nchunks; dt=0.2, desc="channel 2d ", showspeed=true)
for i in 1:nchunks
    t0 = time_ns()
    with_logger(NullLogger()) do
        run!(model, every)
    end
    dt = (time_ns() - t0) * 1e-9
    mlups = Ncell * every / dt / 1e6
    global mlups_ema = isfinite(mlups_ema) ? α * mlups + (1 - α) * mlups_ema : mlups
    export!(model; dir="output_channel_2d")
    next!(prog; showvalues = [
        (:t, Int(d.t)),
        (:t_si, si_t(model.units, Int(d.t))),
        (:MLUPS, round(mlups_ema; digits=1)),
    ])
end
finish!(prog)
