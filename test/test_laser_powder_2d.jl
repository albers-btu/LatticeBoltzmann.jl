using Test
using LatticeBoltzmann

@testset "2D powder bed refuses to paint" begin
    @test DIM == 2
    bed = PowderBed(zeros(Float32, 2, 2, 2), nothing, 1.0, 1.0, 1.0, (0.0, 0.0, 0.0))
    flags = fill(TYPE_S, 8)
    Tfield = fill(0.2f0, 8)
    fs = fill(0.3f0, 8)
    flags0 = copy(flags)
    T0 = copy(Tfield)
    fs0 = copy(fs)
    @test_throws ArgumentError paint_powder_bed!(flags, Tfield, fs, bed, 0.2f0)
    @test flags == flags0
    @test Tfield == T0
    @test fs == fs0
end

@testset "2D plic_hit 9-neighborhood" begin
    # +x full, −x empty: metal→gas normal points toward −x and has nz = 0.
    phij = (0.5f0, 1.0f0, 0.0f0, 0.5f0, 0.5f0, 1.0f0, 0.0f0, 1.0f0, 0.0f0)
    @test phij isa NTuple{9,Float32}
    hit, t, nϕ = LatticeBoltzmann.plic_hit(
        0.5f0, phij,
        0.2f0, 1.0f0, 1.0f0,
        1.0f0, 0.0f0, 0.0f0,
        1.0f0, 1.0f0, 1.0f0,
    )
    @test hit
    @test t ≈ 0.8f0 atol=1f-5
    @test nϕ ≈ Float32[-1, 0, 0]

    # A 27-tuple still hits the untyped-phij method.
    phij27 = ntuple(_ -> 0.0f0, 27)
    hit27, t27, _n27 = LatticeBoltzmann.plic_hit(
        0.5f0, phij27,
        0.0f0, 0.0f0, 1.0f0,
        0.0f0, 0.0f0, -1.0f0,
        1.0f0, 1.0f0, 1.0f0,
    )
    @test hit27 isa Bool
    @test t27 isa Float32
end

@testset "2D walks stay on z = 1" begin
    Nx, Ny, Nz = 4, 4, 1
    N = Nx * Ny * Nz
    flags = zeros(UInt8, N)
    flags[2 + (2 - 1) * Nx] = TYPE_I
    ϕ = zeros(Float32, N)
    Q = zeros(Float32, N)
    path = NTuple{4,Float64}[]
    LatticeBoltzmann._walk_laser_ray!(
        Q, flags, ϕ,
        2.0f0, 2.0f0, 4.0f0, 1.0f0, 0.0f0, -1.0f0, 1.0f0,
        3.27f0, 4.48f0, 4, 1, 1.0f0,
        Nx, Ny, Nz, path,
    )
    @test !isempty(path)
    @test all(p -> p[3] == 1.0, path)
    @test path[end][1] > path[1][1]
    @test all(iszero, Q)

    pathv = NTuple{4,Float64}[]
    LatticeBoltzmann._walk_laser_ray!(
        Q, flags, ϕ,
        2.0f0, 2.0f0, 4.0f0, 0.0f0, 0.0f0, -1.0f0, 1.0f0,
        3.27f0, 4.48f0, 4, 1, 1.0f0,
        Nx, Ny, Nz, pathv,
    )
    @test isempty(pathv)

    mp = zeros(Float32, N)
    mass = zeros(Float32, N)
    ox, oy, oz, live, dm = LatticeBoltzmann._walk_parcel!(
        mp, mass, flags, ϕ, 0.0f0,
        2.0f0, 2.0f0, 4.0f0, 0.0f0, -1.0f0, -1.0f0,
        10.0f0, 0.25f0, Nx, Ny, Nz,
    )
    @test oz == 1.0f0
    @test oy < 2.0f0
    @test live == false
    @test dm == 0

    ox, oy, oz, live, dm = LatticeBoltzmann._walk_parcel!(
        mp, mass, flags, ϕ, 0.0f0,
        2.0f0, 2.0f0, 4.0f0, 0.0f0, 0.0f0, -1.0f0,
        10.0f0, 0.25f0, Nx, Ny, Nz,
    )
    @test (ox, oy, oz) == (2.0f0, 2.0f0, 1.0f0)
    @test live == false
    @test dm == 0
end
