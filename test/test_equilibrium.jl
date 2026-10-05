using Test
using LatticeBoltzmann
using KernelAbstractions

@testset "TYPE_E prescribed velocity is held" begin
    @test EQUILIBRIUM_BOUNDARIES
    Nx, Ny, Nz = 12, 8, 8
    ux0 = 0.04f0
    model = Model(Nx, Ny, Nz, 0.02; backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    uh = zeros(Float32, Nx * Ny * Nz, 3)
    ρh = ones(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        if x == 1
            host[n] = TYPE_E
            uh[n, 1] = ux0
        elseif x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
            host[n] = TYPE_S
        else
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].u.data, uh)
    copyto!(model.domains[1].ρ.data, ρh)
    initialize!(model)
    run!(model, 20)
    moments!(model)
    u = Array(model.domains[1].u.data)
    flags = Array(model.domains[1].flags.data)
    inlet = (flags .& TYPE_BO) .== TYPE_E
    @test count(inlet) == Ny * Nz
    @test all(isapprox.(u[inlet, 1], ux0; atol=1.0f-6))
    @test all(isapprox.(u[inlet, 2], 0; atol=1.0f-6))
    @test all(isapprox.(u[inlet, 3], 0; atol=1.0f-6))
end

@testset "EsotericPull init keeps the sign of a uniform velocity" begin
    Nx, Ny, Nz = 8, 8, 8
    ux0 = 0.05f0
    model = Model(Nx, Ny, Nz, 0.02; fx=0.0f0, fy=0.0f0, fz=0.0f0, β=0.0f0,
                  backend=CPU(), workgroup=64)
    N = Nx * Ny * Nz
    copyto!(model.domains[1].flags.data, fill(TYPE_F, N))
    uh = zeros(Float32, N, 3)
    uh[:, 1] .= ux0
    copyto!(model.domains[1].u.data, uh)
    copyto!(model.domains[1].ρ.data, ones(Float32, N))
    initialize!(model)
    moments!(model)
    u0 = Array(model.domains[1].u.data)
    @test all(isapprox.(u0[:, 1], ux0; atol=1.0f-5))
    @test all(isapprox.(u0[:, 2], 0; atol=1.0f-5))
    run!(model, 2)
    moments!(model)
    u2 = Array(model.domains[1].u.data)
    @test all(isapprox.(u2[:, 1], ux0; atol=1.0f-4))
end
