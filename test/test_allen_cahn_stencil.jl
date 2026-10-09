using Test
using LatticeBoltzmann

@testset "Allen-Cahn stencil on polynomials" begin
    w = weights(SCHEME, Float64)
    c = velocities(SCHEME)
    Nx, Ny, Nz = DIM == 2 ? (5, 5, 1) : (5, 5, 5)
    N = Nx * Ny * Nz
    lin = zeros(Float64, N)
    quad = zeros(Float64, N)
    az = DIM == 2 ? 0.0 : 3.0
    for z in 0:Nz-1, y in 0:Ny-1, x in 0:Nx-1
        n = x + y * Nx + z * Nx * Ny + 1
        lin[n] = x + 2.0 * y + az * z
        quad[n] = Float64(x)^2
    end
    x, y = 2, 2
    z = DIM == 2 ? 0 : 2
    gx, gy, gz = LatticeBoltzmann.grad_phi(lin, x, y, z, w, c, Nx, Ny, Nz)
    @test gx ≈ 1 atol=1e-12
    @test gy ≈ 2 atol=1e-12
    @test gz ≈ az atol=1e-12
    @test LatticeBoltzmann.laplacian_phi(lin, x, y, z, w, c, Nx, Ny, Nz) ≈ 0 atol=1e-12
    gx, gy, gz = LatticeBoltzmann.grad_phi(quad, x, y, z, w, c, Nx, Ny, Nz)
    @test gx ≈ 2x atol=1e-12
    @test gy ≈ 0 atol=1e-12
    @test gz ≈ 0 atol=1e-12
    @test LatticeBoltzmann.laplacian_phi(quad, x, y, z, w, c, Nx, Ny, Nz) ≈ 2 atol=1e-12
end

@testset "Allen-Cahn equilibrium profile" begin
    W = 4.0
    σ = 1.0
    w = weights(SCHEME, Float64)
    c = velocities(SCHEME)
    if DIM == 2
        Nx, Ny, Nz = 8, 1, 1
    else
        Nx, Ny, Nz = 1, 1, 8
    end
    phi = zeros(Float64, Nx * Ny * Nz)
    for z in 0:Nz-1, y in 0:Ny-1, x in 0:Nx-1
        s = DIM == 2 ? (x - Nx / 2) : (z - Nz / 2)
        n = x + y * Nx + z * Nx * Ny + 1
        phi[n] = (1 + tanh(2 * s / W)) / 2
    end

    mid_mag = NaN
    mid_target = NaN
    mu_max = 0.0
    for z in 0:Nz-1, y in 0:Ny-1, x in 0:Nx-1
        n = x + y * Nx + z * Nx * Ny + 1
        s = DIM == 2 ? (x - Nx / 2) : (z - Nz / 2)
        gx, gy, gz = LatticeBoltzmann.grad_phi(phi, x, y, z, w, c, Nx, Ny, Nz)
        lap = LatticeBoltzmann.laplacian_phi(phi, x, y, z, w, c, Nx, Ny, Nz)
        μ = LatticeBoltzmann.mu_phi(phi[n], lap, σ, W)
        mag = sqrt(gx * gx + gy * gy + gz * gz)
        if DIM == 2
            @test gy ≈ 0 atol=1e-12
            @test gz ≈ 0 atol=1e-12
            seam = x == 0 || x == Nx - 1
        else
            @test gx ≈ 0 atol=1e-12
            @test gy ≈ 0 atol=1e-12
            seam = z == 0 || z == Nz - 1
        end
        if s == 0
            mid_mag = mag
            mid_target = 4 * phi[n] * (1 - phi[n]) / W
            @test μ ≈ 0 atol=1e-12
        end
        seam || (mu_max = max(mu_max, abs(μ)))
    end
    # Central difference of tanh(2s/W) at W = 4 misses |∇φ| by ~0.019
    # and leaves |μ| ~ 0.057 σ. 1e-3 is below that truncation.
    @test abs(mid_mag - mid_target) < 2e-2
    @test mu_max < 6e-2 * σ
end

@testset "mu_phi formula" begin
    phi = 0.2
    lap = -0.05
    σ = 0.3
    W = 4.0
    expect = 1.5 * σ * ((16 / W) * phi * (1 - phi) * (1 - 2 * phi) - W * lap)
    @test LatticeBoltzmann.mu_phi(phi, lap, σ, W) ≈ expect atol=1e-15
end
