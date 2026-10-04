using Test
using LatticeBoltzmann

# Zero-based index, same layout as src_index. Cell (4, 2, 0) on 10×8×1 is n = 25.
@inline function n_of(x, y, z, Nx, Ny)
    return x + y * Nx + z * Nx * Ny + 1
end

@testset "2D marangoni and recoil drop z" begin
    @test DIM == 2

    Nx, Ny, Nz = 10, 8, 1
    N = Nx * Ny * Nz
    Tfield = Vector{Float64}(undef, N)
    ϕ = Vector{Float64}(undef, N)
    flags = fill(TYPE_F, N)
    for z in 0:(Nz - 1), y in 0:(Ny - 1), x in 0:(Nx - 1)
        n = n_of(x, y, z, Nx, Ny)
        # ∇T = (1/2, 1/4, 0), ∇ϕ = (3, 4, 0). No z slope to sample.
        Tfield[n] = 0.5 * x + 0.25 * y + 10.0
        ϕ[n] = 3.0 * x + 4.0 * y
    end

    x, y, z = 4, 2, 0
    n = n_of(x, y, z, Nx, Ny)
    @test n == 25
    @test LatticeBoltzmann.src_index(x, y, z, 0, 0, 0, Nx, Ny, Nz) == n
    # ±z wraps onto this cell. The 2D arm must not load that offset.
    @test LatticeBoltzmann.src_index(x, y, z, 0, 0, 1, Nx, Ny, Nz) == n
    @test LatticeBoltzmann.src_index(x, y, z, 0, 0, -1, Nx, Ny, Nz) == n

    # n̂ = ∇ϕ/|∇ϕ| = (3, 4, 0)/5
    # n̂·∇T = (3/5)*(1/2) + (4/5)*(1/4) = 1/2
    # ∇T − n̂(n̂·∇T) = (1/2 − 3/10, 1/4 − 2/5, 0) = (1/5, −3/20, 0)
    # F = σT * that, σT = 2 → (2/5, −3/10, 0) = (0.4, −0.3, 0)
    σT = 2.0
    fx, fy, fz = LatticeBoltzmann.marangoni_force(
        Tfield, ϕ, flags, σT, x, y, z, n, Nx, Ny, Nz, Float64)
    @test fx ≈ 0.4 atol=1e-12
    @test fy ≈ -0.3 atol=1e-12
    @test fz == 0.0

    # T = T_v ⇒ p_sat = p0 = 1, p_r = min(0.54, 0.5) = 1/2
    # F = p_r n̂ = (1/2)*(3/5, 4/5, 0) = (0.3, 0.4, 0)
    fill!(Tfield, 2.0)
    rx, ry, rz = LatticeBoltzmann.recoil_force(
        Tfield, ϕ, n, x, y, z, Nx, Ny, Nz, 1.0, 2.0, 1.0, 12.0, Float64)
    @test rx ≈ 0.3 atol=1e-12
    @test ry ≈ 0.4 atol=1e-12
    @test rz == 0.0

    fill!(Tfield, 2.0)
    fill!(ϕ, 0.5)
    fx, fy, fz = LatticeBoltzmann.marangoni_force(
        Tfield, ϕ, flags, σT, x, y, z, n, Nx, Ny, Nz, Float64)
    @test (fx, fy, fz) == (0.0, 0.0, 0.0)
    rx, ry, rz = LatticeBoltzmann.recoil_force(
        Tfield, ϕ, n, x, y, z, Nx, Ny, Nz, 1.0, 2.0, 1.0, 12.0, Float64)
    @test (rx, ry, rz) == (0.0, 0.0, 0.0)
end
