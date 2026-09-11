# Von Kármán vortex street behind a circular cylinder.
# Same setup as flow_past_cylinder.jl; Re and domain chosen so the wake
# sheds (onset ~Re 47; St ≈ 0.16–0.18 at Re 100). Watch Cl oscillate.
#
#   inlet/outlet (x=1, x=Nx) -> TYPE_E, u=(u_in,0,0)
#   cylinder                 -> TYPE_S, drag/lift from update_force_field!
#   z-walls                  -> bounce-back; y is periodic (2D cylinder)
#
#   FORCE_FIELD            = true
#   EQUILIBRIUM_BOUNDARIES = true
#   TRT                    = true   # recommended at Re 100

using LatticeBoltzmann
using Printf
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert FORCE_FIELD && EQUILIBRIUM_BOUNDARIES

Nx, Ny, Nz = 192, 8, 96             # ~8 D of wake, H/D = 6
si_L = 0.384u"m"
si_H = Nz / Nx * si_L
si_D = 0.032u"m"                    # 16 lattice cells
si_u = 0.05u"m/s"
si_ρ = 1.204u"kg/m^3"
Re   = 100
ν    = si_u * si_D / Re

Ma = 0.08
cs = 1 / sqrt(3)
lbm_inlet = Ma * cs

units = Units(si_L, si_u, si_ρ; x=Nx, u=lbm_inlet, ρ=1, T=Float32)
D_cells = ustrip(u"m", si_D) / units.m
τ = 3 * lbm_ν(units, ν) + 0.5
@info "vortex shedding" Re ν Ma D_cells blockage=D_cells/Nz τ

model = Model(Nx, Ny, Nz, units; ν = ν, gx = 0, gy = 0, gz = 0, backend = CUDABackend())

u_in = Float32(lbm_u(units, si_u))
cx = 3.5 * D_cells
cz = (Nz + 1) / 2 + 1.0             # 1-cell offset: breaks mirror symmetry
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
        # small cross-flow seed on the inlet only (not the outlet)
        if x == 1
            uh[n, 3] = 0.02f0 * u_in * sin(Float32(2π * (z - cz) / Nz))
        end
    elseif z == 1 || z == Nz
        host[n] = TYPE_S
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].u.data, uh)
copyto!(model.domains[1].ρ.data, ρh)

d = model.domains[1]
LatticeBoltzmann.initialize!(model)
export!(model; dir="output_vortex")

# ~80 D/U; street often appears after 30–50 D/U from a near-symmetric start.
nsteps = 30000
every  = 200
nchunks = nsteps ÷ every
A = D_cells * Ny
Ncell = Int(model.Nx) * Int(model.Ny) * Int(model.Nz)
mlups_ema = NaN
α = 0.2
prog = Progress(nchunks; dt=0.2, desc="vortex shedding ", showspeed=true)
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
    Fh = Array(d.F.data)
    Fx = sum(Fh[cyl, 1])
    Fz = sum(Fh[cyl, 3])
    Cd = 2 * Fx / (u_in^2 * A)
    Cl = 2 * Fz / (u_in^2 * A)
    export!(model; dir="output_vortex")
    next!(prog; showvalues = [
        (:t, Int(d.t)),
        (:t_si, si_t(model.units, Int(d.t))),
        (:MLUPS, round(mlups_ema; digits=1)),
        (:Cd, round(Cd; digits=3)),
        (:Cl, round(Cl; digits=3)),
    ])
end
finish!(prog)
