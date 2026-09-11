# Velocity-driven duct (wind tunnel). Classic TYPE_E use:
#   x = 1 and x = Nx -> equilibrium BC, prescribed ρ=1, u=(u_in,0,0)
#   other faces      -> bounce-back walls
#   EQUILIBRIUM_BOUNDARIES = true
using LatticeBoltzmann
using Printf
using CUDA
using Unitful

@assert EQUILIBRIUM_BOUNDARIES

Nx, Ny, Nz = 48, 24, 24
si_L = 0.048u"m"                    # streamwise length (cubic cells)
si_H = Nz / Nx * si_L               # duct height
si_W = Ny / Nx * si_L

# Air at 20 °C. Water at this size/speed is Re~2e3 and ω->2 on 48×24×24.
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
    elseif y == 1 || y == Ny || z == 1 || z == Nz
        host[n] = TYPE_S
    else
        host[n] = TYPE_F
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].u.data, uh)
copyto!(model.domains[1].ρ.data, ρh)

d = model.domains[1]
LatticeBoltzmann.initialize!(model)
export!(model; dir="output_channel")

nsteps = 2000
every  = 50
for i in 1:(nsteps ÷ every)
    run!(model, every)
    export!(model; dir="output_channel")
    t = Int(d.t)
    @info "dump" t t_si=si_t(model.units, t)*u"s"
end
