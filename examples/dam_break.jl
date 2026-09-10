using LatticeBoltzmann
using Printf
using CUDA
using Unitful

@assert SURFACE && VOLUME_FORCE && UPDATE_FIELDS

Nx, Ny, Nz = 48, 48, 48
si_L = 0.1u"m"                  # tank size
si_H = (2 * Nz ÷ 3) / Nz * si_L # dam height ~ 2/3 box height
si_g = 9.81u"m/s^2"
si_u = sqrt(ustrip(u"m/s^2", si_g) * ustrip(u"m", si_H)) * u"m/s"

Ma = 0.05 # D3Q19 Ma is usual safe below 0.05
cs = 1 / sqrt(3)
lbm_u = Ma * cs

units = Units(si_L, si_u, 1000u"kg/m^3"; x=Nx, u=lbm_u, ρ=1, T=Float32)

# stable SRT band 0.53 … 1
# stable TRT band 0.505 … 1
τ = 0.51f0
ν_lbm = (τ - 0.5f0) / 3
ν = LatticeBoltzmann.si_ν(units, ν_lbm) * u"m^2/s" # 1.2e-3u"m^2/s" # honey at 7.0e-3
@info "Kinematic viscosity $ν"
# ν = 1.0e-6u"m^2/s" # water
σ = 0u"N/m"

model = Model(Nx, Ny, Nz, units;
              ν = ν,                 # kinematic viscosity
              σ = σ,                 # surface tension
              gz = -si_g,            # acceleration due to earth gravity field
              SType=Float16,
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
every  = 20
for i in 1:(nsteps ÷ every)
    run!(model, every)
    export!(model; dir="output")
    @info "dump" t=Int(d.t)
end