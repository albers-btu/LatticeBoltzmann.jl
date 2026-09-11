using Test
using LatticeBoltzmann
using KernelAbstractions

@testset "MOVING_BOUNDARIES lid keeps velocity" begin
    @test MOVING_BOUNDARIES
    Nx, Ny, Nz = 12, 12, 12
    u_lid = 0.05f0
    model = Model(Nx, Ny, Nz, 0.02; backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    uh = zeros(Float32, Nx * Ny * Nz, 3)
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
    LatticeBoltzmann.initialize!(model)
    flags = Array(model.domains[1].flags.data)
    n_mid = 6 + (6 - 1) * Nx + (Nz - 2) * Nx * Ny  # fluid under lid
    @test (flags[n_mid] & TYPE_BO) == TYPE_MS

    run!(model, 20)
    LatticeBoltzmann.moments!(model)
    u = Array(model.domains[1].u.data)
    lid = [z == Nz && x > 1 && x < Nx && y > 1 && y < Ny
           for z in 1:Nz for y in 1:Ny for x in 1:Nx]
    # reshape order: n = x + (y-1)*Nx + (z-1)*Nx*Ny, so iterate x fastest
    lid = falses(Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        if z == Nz && x > 1 && x < Nx && y > 1 && y < Ny
            lid[n] = true
        end
    end
    @test all(isapprox.(u[lid, 1], u_lid; atol=1.0f-6))
    under = (flags .& TYPE_BO) .== TYPE_MS
    @test count(under) > 0
    @test sum(u[under, 1]) / count(under) > 0
end
