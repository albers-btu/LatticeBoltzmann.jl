# 3D lid-driven cavity. Classic MOVING_BOUNDARIES use:
#   z = Nz      -> TYPE_S lid with prescribed u = (u_lid, 0, 0)
#   other faces -> stationary TYPE_S
#   MOVING_BOUNDARIES = true

using LatticeBoltzmann
using Printf
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert MOVING_BOUNDARIES

Nx, Ny, Nz = 48, 48, 48
si_L = 0.1u"m"                      # cube side
si_u = 0.05u"m/s"                   # lid speed
Re   = 100                          # u_lid L / ν
si_ρ = 1000u"kg/m^3"
ν    = si_u * si_L / Re             # 5e-5 m²/s at these defaults

Ma = 0.08
cs = 1 / sqrt(3)
lbm_lid = Ma * cs

units = Units(si_L, si_u, si_ρ; x=Nx, u=lbm_lid, ρ=1, T=Float32)
@info "lid-driven cavity" Re ν si_u si_L τ=(3 * lbm_ν(units, ν) + 0.5)

model = Model(Nx, Ny, Nz, units; ν = ν, gx = 0, gy = 0, gz = 0, backend = CUDABackend())

u_lid = Float32(lbm_u(units, si_u))
host = zeros(UInt8, Nx * Ny * Nz)
uh   = zeros(Float32, Nx * Ny * Nz, 3)
ρh   = ones(Float32, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if z == Nz
        host[n] = TYPE_S
        uh[n, 1] = u_lid
    elseif x == 1 || x == Nx || y == 1 || y == Ny || z == 1
        host[n] = TYPE_S
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].u.data, uh)
copyto!(model.domains[1].ρ.data, ρh)

d = model.domains[1]
LatticeBoltzmann.initialize!(model)
export!(model; dir="output_cavity")

nsteps = 4000
every  = 50
nchunks = nsteps ÷ every
Ncell = Int(model.Nx) * Int(model.Ny) * Int(model.Nz)
mlups_ema = NaN
α = 0.2
prog = Progress(nchunks; dt=0.2, desc="lid-driven cavity ", showspeed=true)
for i in 1:nchunks
    t0 = time_ns()
    with_logger(NullLogger()) do
        run!(model, every)
    end
    dt = (time_ns() - t0) * 1e-9
    mlups = Ncell * every / dt / 1e6
    global mlups_ema = isfinite(mlups_ema) ? α * mlups + (1 - α) * mlups_ema : mlups
    export!(model; dir="output_cavity")
    next!(prog; showvalues = [
        (:t, Int(d.t)),
        (:t_si, si_t(model.units, Int(d.t))),
        (:MLUPS, round(mlups_ema; digits=1)),
    ])
end
finish!(prog)
