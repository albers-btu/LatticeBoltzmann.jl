using Test
using StaticArrays
using LatticeBoltzmann

# Cell (ix, iy) occupies [ix−0.5, ix+0.5]². Liquid is the disk interior.
function _disk_area(ix, iy, cx, cy, R; n=32)
    inside = 0
    h = 1.0 / n
    o = -0.5 + 0.5 * h
    R2 = R * R
    for a in 0:n-1, b in 0:n-1
        x = ix + o + a * h
        y = iy + o + b * h
        if (x - cx)^2 + (y - cy)^2 <= R2
            inside += 1
        end
    end
    return inside / n^2
end

@testset "2D PLIC line offset" begin
    @test DIM == 2
    n = SVector{3,Float32}(0, 1, 0)
    @test LatticeBoltzmann.plic_line(0.5f0, n) ≈ 0 atol=1f-5
    n2 = SVector{3,Float32}(1, 0, 0)
    @test LatticeBoltzmann.plic_line(0.5f0, n2) ≈ 0 atol=1f-5
    n3 = SVector{3,Float32}(1, 1, 0)
    n3 = n3 / sqrt(sum(abs2, n3))
    @test abs(LatticeBoltzmann.plic_line(0.5f0, n3)) < 1f-4
    # empty / full: offset at the far side, and the two sum to 0
    @test LatticeBoltzmann.plic_line(0.0f0, n) < 0
    @test LatticeBoltzmann.plic_line(1.0f0, n) > 0
    @test LatticeBoltzmann.plic_line(0.0f0, n) ≈ -LatticeBoltzmann.plic_line(1.0f0, n) atol=1f-5
end

@testset "2D PLIC flat curvature is ~0" begin
    c = VELOCITIES[:D2Q9]
    phij = ntuple(i -> begin
        cy = Float32(c[i][2])
        cy < 0 ? 1.0f0 : cy > 0 ? 0.0f0 : 0.5f0
    end, 9)
    κ = LatticeBoltzmann.calculate_curvature_2d(phij)
    @test isfinite(κ)
    @test abs(κ) < 0.15f0
end

@testset "2D liquid disk curvature and gas density" begin
    # calculate_curvature on a liquid sphere returns κ < 0 with n metal→gas.
    # The disk uses that shared sign. |κ| near 1/R, not a flipped formula.
    R = 8.0f0
    cx, cy = 12, 12
    Nx, Ny, Nz = 24, 24, 1
    x, y, z = cx + 7, cy + 3, 0
    ϕ = zeros(Float32, Nx * Ny * Nz)
    for iy in 0:(Ny - 1), ix in 0:(Nx - 1)
        ϕ[ix + iy * Nx + 1] = _disk_area(ix, iy, Float64(cx), Float64(cy), Float64(R))
    end
    n = x + y * Nx + 1
    ϕ0 = ϕ[n]
    @test 0 < ϕ0 < 1
    phij = LatticeBoltzmann.gather_phi_d2q9(ϕ, ϕ0, x, y, z, Nx, Ny, Nz)
    κ = LatticeBoltzmann.calculate_curvature_2d(phij)
    @test κ < 0
    @test isapprox(κ, -1 / R; rtol=0.25)
    σ = 0.2f0
    ρg = LatticeBoltzmann.gas_density_plic(σ, ϕ, ϕ0, x, y, z, Nx, Ny, Nz)
    @test ρg == clamp(1 - 3 * σ * κ, 0.2f0, 2.0f0)
end

@testset "2D flat interface step" begin
    Nx, Ny, Nz = 16, 16, 1
    # Nonzero σ so gas_density_plic takes the 2D gather, not the σ = 0 return.
    model = Model(Nx, Ny, Nz, 0.1; fx=0, fy=0, fz=0, σ=0.02)
    N = Nx * Ny * Nz
    @test length(model.domains[1].fi.data) == 9 * N
    host = zeros(UInt8, N)
    for y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx
        # Interior liquid is TYPE_F. Unpainted cells stay 0; SURFACE init makes them gas.
        if 5 <= y <= Ny - 4
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    initialize!(model)
    flags = Array(model.domains[1].flags.data)
    @test any(f -> (f & TYPE_SU) == TYPE_F, flags)
    @test any(f -> (f & TYPE_SU) == TYPE_I, flags)
    for _ in 1:8
        LatticeBoltzmann.step!(model)
    end
    u = Array(model.domains[1].u.data)
    ρ = Array(model.domains[1].ρ.data)
    fi = Array(model.domains[1].fi.data)
    @test all(isfinite, u)
    @test all(isfinite, ρ)
    @test all(isfinite, fi)
    @test all(==(0), @view u[:, 3])
end
