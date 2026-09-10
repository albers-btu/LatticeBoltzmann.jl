using Test
using StaticArrays
using LatticeBoltzmann

@testset "PLIC cube offset" begin
    n = SVector{3,Float32}(0, 0, 1)
    @test plic_cube(0.5f0, n) ≈ 0 atol=1f-5
    n2 = SVector{3,Float32}(1, 0, 0)
    @test plic_cube(0.5f0, n2) ≈ 0 atol=1f-5
    n3 = SVector{3,Float32}(1, 1, 1)
    n3 = n3 / sqrt(sum(abs2, n3))
    @test abs(plic_cube(0.5f0, n3)) < 1f-4
    # empty / full cube: offset at the far face, |d| ~ L1(n)/2
    @test plic_cube(0.0f0, n) < 0
    @test plic_cube(1.0f0, n) > 0
    @test plic_cube(0.0f0, n) ≈ -plic_cube(1.0f0, n) atol=1f-5
end

@testset "PLIC plane curvature is ~0" begin
    # z-normal plane through the cell: below +z is liquid, above is gas
    phij = ntuple(Val(27)) do i
        cz = Float32(LatticeBoltzmann.D3Q27_C[i][3])
        cz < 0 ? 1.0f0 : cz > 0 ? 0.0f0 : 0.5f0
    end
    κ = calculate_curvature(phij)
    @test isfinite(κ)
    @test abs(κ) < 0.15f0
end

@testset "PLIC σ=0 leaves ρ_gas=1" begin
    ϕ = fill(0.5f0, 5 * 5 * 5)
    ρg = LatticeBoltzmann.gas_density_plic(0.0f0, ϕ, 0.5f0, 2, 2, 2, 5, 5, 5)
    @test ρg == 1.0f0
end
