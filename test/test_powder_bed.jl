using Test
using LatticeBoltzmann

@testset "powder bed HDF5" begin
    path = joinpath(pkgdir(LatticeBoltzmann), "input", "powder_bed.h5")
    @test isfile(path)
    bed = read_powder_bed(path)
    Nx, Ny, Nz = size(bed.phi)
    @test (Nx, Ny, Nz) == (435, 109, 72)
    @test bed.dx ≈ 4.911264239267441e-6
    @test bed.origin[1] == 0 && bed.origin[2] == 0
    @test all(0 .<= bed.phi .<= 1)
    @test bed.substrate !== nothing
    @test substrate_top(bed) == 41
    @test count(>(0), bed.phi) > 0
    @test count(>(0), @view bed.phi[:, :, 1]) == 0
    @test count(>(0), @view bed.substrate[:, :, 1]) == Nx * Ny

    flags = zeros(UInt8, Nx * Ny * (Nz + 3))
    T = fill(0.2f0, length(flags))
    fs = zeros(Float32, length(flags))
    Hfill = paint_powder_bed!(flags, T, fs, bed, 0.2f0; z0=2)
    @test Hfill == 42
    n_plate = 1 + (Ny ÷ 2 - 1) * Nx + (Hfill - 1) * Nx * Ny
    @test flags[n_plate] == TYPE_F
    @test fs[n_plate] == 1
    # a gas cell above the plate that the file left empty stays empty
    n_hi = 1 + (2 - 1) * Nx + (2 + Nz - 1) * Nx * Ny
    @test flags[n_hi] == 0x00 || flags[n_hi] == TYPE_F
end
