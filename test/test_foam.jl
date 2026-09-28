using Test, LatticeBoltzmann
using KernelAbstractions: CPU

@testset "foam allocation and no-op step" begin
    @test FOAM

    model = Model(8, 8, 8, 0.1; backend=CPU())
    domain = model.domains[1]
    N = length(domain.flags)
    @test hasproperty(domain, :ci)
    @test hasproperty(domain, :tag)
    @test length(domain.ci) == length(domain.gi) == 7 * N
    @test length(domain.c) == N
    @test length(domain.ϕ_old) == N
    @test length(domain.ρb) == N
    @test length(domain.Pi) == N
    @test length(domain.tag) == N
    @test eltype(domain.tag) == Int32
    @test length(domain.flux) == 4096
    @test all(iszero, Array(domain.ϕ_old.data))
    @test domain.D == 0
    @test domain.γ_b == 1
    @test domain.ρ_liquid == 1

    set_foam!(model; D=0.03, k_H=0.001, k_Π=0, q=0, V_m=0, γ_b=1, c0=0, ρ_liquid=1)
    @test domain.D == Float32(0.03)
    @test domain.k_H == Float32(0.001)
    stats = bubble_stats(model)
    @test stats.n == 0
    @test stats.ΣV == 0
    @test stats.mean_ratio == 0
    @test stats.max_abs_ρb == 1
    @test stats.max_Π == 0
    @test !hasproperty(stats, :clamp_hits)

    fill!(domain.flags.data, TYPE_F)
    initialize!(model)
    LatticeBoltzmann.step!(model)
    for arr in (domain.ρ.data, domain.u.data, domain.fi.data, domain.gi.data, domain.T.data,
                domain.ci.data, domain.c.data, domain.ϕ_old.data, domain.ρb.data, domain.Pi.data)
        @test all(isfinite, Array(arr))
    end
    @test all(iszero, Array(domain.ϕ_old.data))

    model2 = Model(4, 4, 4, 0.1; backend=CPU(), n_hydro=2)
    @test_throws "FOAM v1 requires n_hydro == 1 (got 2); substep scaling is not applied" LatticeBoltzmann.step!(model2)
end
