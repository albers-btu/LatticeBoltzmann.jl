using Test
using LatticeBoltzmann

# Direct map. Compiled only when this file is included (DIM == 2).
# χ = cs² (1/ω - 1/2), cs² = 1/3, domain.α = 2χ ⇒ 1/ω = (3/2)α + 1/2.
@testset "D2Q5 omega" begin
    @test DIM == 2
    α = 0.02
    ω = LatticeBoltzmann.omega_T_from_alpha(α)
    @test ω == 1 / (1.5 * α + 0.5)
    @test ω == 1 / 0.53
    k = LatticeBoltzmann.thermal_conductivity(ω)
    # (1/3)*(1/ω - 1/2) equals α/2 in exact arithmetic. The inversion
    # 1/ω is one rounding off that identity, so this is ≈, not ==.
    @test k ≈ α / 2
    # D3Q7's 0.25*(1/ω - 1/2) is 0.375α, not α/2.
    @test !isapprox(k, 0.25 * (1 / ω - 0.5); rtol=1e-3)

    α32 = 0.02f0
    model = Model(4, 4, 1, 0.1; α=α32, σ=0.0, fx=0.0, fy=0.0, fz=0.0)
    d = model.domains[1]
    ω32 = LatticeBoltzmann.omega_T_from_alpha(α32)
    @test ω32 == 1 / (1.5f0 * α32 + 0.5f0)
    @test d.ω_T == ω32
    @test LatticeBoltzmann.thermal_k(d) ≈ α32 / 2
    # thermal_k_s / thermal_k_l stay 0.5*α and do not go through cs².
    @test LatticeBoltzmann.thermal_k_s(d) == 0.5f0 * d.α_s
    @test LatticeBoltzmann.thermal_k_l(d) == 0.5f0 * d.α_l
end

# store_geq_local! writes one population per index. Pairs 2 and 4 alone
# would leave populations 3 and 5 at the allocation fill. At T = 1, geq is
# 0, so a wall at Tm hides that hole. Check a non-unit wall temperature.
@testset "D2Q5 solid geq fills populations 2:5" begin
    N = 1
    n = 1
    Twall = 1.3
    gi = zeros(5 * N)
    LatticeBoltzmann.store_geq_local!(gi, n, Twall, N)
    g0 = (Twall - 1) / 3
    gax = (Twall - 1) / 6
    @test gax != 0
    @test gi[LatticeBoltzmann.f_index(n, 1, N)] ≈ g0
    for i in 2:5
        @test gi[LatticeBoltzmann.f_index(n, i, N)] ≈ gax
    end
end

# Body-force channel. Walls at y = 1 and y = Ny, periodic in x.
# Halfway bounce-back: H = Ny - 2. Do not compare the node profile;
# s = i - 1/2, and this test only locks u_max, |uy|, and |uz|.
@testset "2D Poiseuille" begin
    Nx, Ny, Nz = 64, 32, 1
    ν = 0.1
    H = Ny - 2
    @test H == 30
    fx = 0.04 / (900 * sqrt(3))
    u_max = fx * H^2 / (8ν)
    @test u_max ≈ 0.05 / sqrt(3)

    model = Model(Nx, Ny, Nz, ν; fx=fx, fy=0, fz=0, σ=0)
    host = zeros(UInt8, Nx * Ny * Nz)
    for y in 1:Ny, x in 1:Nx
        z = 1
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        host[n] = (y == 1 || y == Ny) ? TYPE_S : TYPE_F
    end
    copyto!(model.domains[1].flags.data, host)
    initialize!(model)
    # ν = 0.1, H = 30: the slowest viscous mode is ~ν π²/H².
    # A few thousand steps puts the centerline inside 2%.
    for _ in 1:6000
        LatticeBoltzmann.step!(model)
    end
    u = Array(model.domains[1].u.data)
    @test all(isfinite, u)
    measured = maximum(abs, @view u[:, 1])
    uy_max = maximum(abs, @view u[:, 2])
    uz_max = maximum(abs, @view u[:, 3])
    @info "2D Poiseuille" measured u_max uy_max uz_max ratio = measured / u_max
    @test isapprox(measured, u_max; rtol=0.02)
    @test uy_max < 1e-4
    @test uz_max == 0
end
