using LatticeBoltzmann
using Printf
using CUDA

@assert SURFACE && VOLUME_FORCE && UPDATE_FIELDS

Nx, Ny, Nz = 64, 64, 64
ν = 0.008f0
fz = -8.0f-4

model = Model(Nx, Ny, Nz, ν; fz=fz, SType=Float16, backend=CUDABackend())


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

LatticeBoltzmann.initialize!(model)

d = model.domains[1]
su = d.flags.data .& TYPE_SU
@show count(==(TYPE_F), su) count(==(TYPE_I), su) count(==(TYPE_G), su)
@show extrema(model.domains[1].ϕ.data)
@info "init" S=count(f -> (f & TYPE_S) == TYPE_S, d.flags.data) F=count(==(TYPE_F), su) I=count(==(TYPE_I), su) G=count(==(TYPE_G), su) mass=sum(d.mass.data)

export!(model; dir="output")   # t=0, writes output/lbm_00000000.vti + output/lbm.pvd

nsteps = 2000
every  = 20
mass0  = sum(d.mass.data)
for i in 1:(nsteps ÷ every)
    run!(model, every)
    export!(model; dir="output")
    m = sum(d.mass.data)
    @info "dump" t=Int(d.t) mass=m rel=(m - mass0) / mass0
end