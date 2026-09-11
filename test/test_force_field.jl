using Test
using LatticeBoltzmann
using KernelAbstractions

@testset "FORCE_FIELD body force accelerates fluid" begin
    @test FORCE_FIELD
    Nx = Ny = Nz = 12
    Fz = 1.0f-4
    nsteps = 40
    model = Model(Nx, Ny, Nz, 0.02; backend=CPU(), workgroup=64)
    Fd = model.domains[1].F.data
    Fd[:, 3] .= Fz
    LatticeBoltzmann.initialize!(model)
    run!(model, nsteps)
    LatticeBoltzmann.moments!(model)
    uz = Array(model.domains[1].u.data)[:, 3]
    # Guo: Δ(ρu) = F per step, ρ≈1 → uz ≈ nsteps * Fz
    @test isapprox(sum(uz) / length(uz), nsteps * Fz; rtol=0.15)
end

@testset "FORCE_FIELD momentum exchange on solids" begin
    @test FORCE_FIELD
    Nx, Ny, Nz = 12, 8, 8
    model = Model(Nx, Ny, Nz, 0.02; backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        if z == 1 || z == Nz
            host[n] = TYPE_S
        end
    end
    copyto!(model.domains[1].flags.data, host)
    model.domains[1].F.data[:, 1] .= 2.0f-4
    LatticeBoltzmann.initialize!(model)
    run!(model, 30)
    reset_force_field!(model)
    update_force_field!(model)
    Fh = Array(model.domains[1].F.data)
    flags = Array(model.domains[1].flags.data)
    solids = (flags .& TYPE_BO) .== TYPE_S
    fluid = .!solids
    @test any(solids)
    @test all(iszero, Fh[fluid, :])
    @test any(!iszero, Fh[solids, 1])
end
