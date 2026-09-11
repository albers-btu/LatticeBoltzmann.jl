# Flow past a circular cylinder (von Kármán setup). Classic FORCE_FIELD use:
#   inlet/outlet (x=1, x=Nx) → TYPE_E, u=(u_in,0,0)
#   cylinder                 → TYPE_S, drag from update_force_field!
#   other faces              → bounce-back
#
#   TRT                    = true   # nicer at Re ≳ 50
#   FORCE_FIELD            = true
#   EQUILIBRIUM_BOUNDARIES = true

using LatticeBoltzmann
using Printf
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert FORCE_FIELD && EQUILIBRIUM_BOUNDARIES
start_run_log!("output_cylinder")

Nx, Ny, Nz = 96, 8, 48
si_L = 0.192u"m"
si_H = Nz / Nx * si_L               # 0.096 m, H/D = 6
si_D = 0.016u"m"                    # 8 lattice cells
si_u = 0.04u"m/s"
si_ρ = 1.204u"kg/m^3"
Re   = 20
ν    = si_u * si_D / Re

Ma = 0.05
cs = 1 / sqrt(3)
lbm_inlet = Ma * cs

units = Units(si_L, si_u, si_ρ; x=Nx, u=lbm_inlet, ρ=1, T=Float32)
D_cells = ustrip(u"m", si_D) / units.m
τ = 3 * lbm_ν(units, ν) + 0.5
@info "cylinder" Re ν Ma D_cells blockage=D_cells/Nz τ

model = Model(Nx, Ny, Nz, units; ν = ν, gx = 0, gy = 0, gz = 0, backend = CUDABackend())

u_in = Float32(lbm_u(units, si_u))
cx = 2 * D_cells                    # two diameters from inlet
cz = (Nz + 1) / 2
R  = D_cells / 2

host = zeros(UInt8, Nx * Ny * Nz)
uh   = zeros(Float32, Nx * Ny * Nz, 3)
ρh   = ones(Float32, Nx * Ny * Nz)
cyl  = falses(Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if (x - cx)^2 + (z - cz)^2 <= R^2
        host[n] = TYPE_S
        cyl[n] = true
    elseif x == 1 || x == Nx
        host[n] = TYPE_E
        uh[n, 1] = u_in
    elseif z == 1 || z == Nz
        host[n] = TYPE_S
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].u.data, uh)
copyto!(model.domains[1].ρ.data, ρh)

d = model.domains[1]
LatticeBoltzmann.initialize!(model)
export!(model; dir="output_cylinder")

nsteps = 4000
every  = 50
nchunks = nsteps ÷ every
A = D_cells * Ny                         # projected area (span × diameter)
Ncell = Int(model.Nx) * Int(model.Ny) * Int(model.Nz)
mlups_ema = NaN
α = 0.2
prog = Progress(nchunks; dt=0.2, desc="flow past cylinder ", showspeed=true)
for i in 1:nchunks
    t0 = time_ns()
    with_logger(NullLogger()) do
        run!(model, every)
    end
    dt = (time_ns() - t0) * 1e-9
    mlups = Ncell * every / dt / 1e6
    global mlups_ema = isfinite(mlups_ema) ? α * mlups + (1 - α) * mlups_ema : mlups
    reset_force_field!(model)
    update_force_field!(model)
    Fx = sum(Array(d.F.data)[cyl, 1])
    Cd = 2 * Fx / (u_in^2 * A)
    export!(model; dir="output_cylinder")
    next!(prog; showvalues = [
        (:t, Int(d.t)),
        (:t_si, si_t(model.units, Int(d.t))),
        (:MLUPS, round(mlups_ema; digits=1)),
        (:Cd, round(Cd; digits=3)),
    ])
end
finish!(prog)
