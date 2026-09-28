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

function _no_surface_transition(flags)
    for f in flags
        su = f & TYPE_SU
        (su == TYPE_IF || su == TYPE_IG || su == TYPE_GI) && return false
    end
    return true
end

@testset "poisson disk nuclei stay one id each" begin
    N = 32
    R = 3.0
    rmin = 2R + 1
    n = 4
    pts = poisson_disk_centers(N, N, N, rmin, n; seed=1)
    @test length(pts) == n
    for i in 1:n, j in 1:(i - 1)
        dx = pts[i][1] - pts[j][1]
        dy = pts[i][2] - pts[j][2]
        dz = pts[i][3] - pts[j][3]
        dx -= N * round(dx / N)
        dy -= N * round(dy / N)
        dz -= N * round(dz / N)
        @test dx * dx + dy * dy + dz * dz >= rmin * rmin - 1e-8
    end
    model = Model(N, N, N, 0.1; backend=CPU(), σ=0)
    domain = model.domains[1]
    fill!(domain.flags.data, TYPE_F)
    nucleate_bubbles!(model, pts, fill(R, n))
    @test bubble_count(model) == n
    initialize!(model)
    LatticeBoltzmann.step!(model)
    ids = bubble_ids(model)
    @test length(ids) == n
    @test length(unique(ids)) == n
end

@testset "one punched sphere keeps its id" begin
    N = 32
    R = 6.0
    model = Model(N, N, N, 0.1; backend=CPU(), σ=0)
    set_foam!(model; D=0)
    domain = model.domains[1]
    fill!(domain.flags.data, TYPE_F)
    nucleate_bubbles!(model, [(N / 2, N / 2, N / 2)], [R])
    @test bubble_count(model) == 1
    id = only(bubble_ids(model))
    @test bubble_ratio(model, id) == 1
    # Cap volume, before the free surface moves ϕ. A couple percent is the
    # plane-cap error, not the later hydrodynamic drift at σ = 0.
    Vanal = 4 / 3 * π * R^3
    V0 = bubble_volume(model, id)
    @test abs(V0 - Vanal) / Vanal < 0.02

    flags0 = Array(domain.flags.data)
    ϕ0 = Array(domain.ϕ.data)
    i_shell = findfirst(i -> (flags0[i] & TYPE_SU) == TYPE_I && 0 < ϕ0[i] < 1, eachindex(flags0))
    @test i_shell !== nothing
    ϕ_shell = ϕ0[i_shell]
    Vref0 = model.foam.bubbles[Int(id)].V_ref
    @test Vref0 == V0

    initialize!(model)
    flags1 = Array(domain.flags.data)
    ϕ1 = Array(domain.ϕ.data)
    @test (flags1[i_shell] & TYPE_SU) == TYPE_I
    @test ϕ1[i_shell] == ϕ_shell
    @test bubble_volume(model, id) == V0

    for _ in 1:20
        LatticeBoltzmann.step!(model)
    end
    @test bubble_count(model) == 1
    @test bubble_ids(model) == [id]
    @test bubble_ratio(model, id) == 1
    @test model.foam.bubbles[Int(id)].V_ref == Vref0
    ρb = Array(domain.ρb.data)
    @test all(==(one(eltype(ρb))), ρb)
    # The fill at the start of the last step read the previous surface_3.
    @test _no_surface_transition(model.foam.flags)
end

@testset "two far spheres stay two ids" begin
    N = 32
    R = 3.5
    model = Model(N, N, N, 0.1; backend=CPU(), σ=0)
    domain = model.domains[1]
    fill!(domain.flags.data, TYPE_F)
    nucleate_bubbles!(model, [(8.5, 16.0, 16.0), (24.5, 16.0, 16.0)], [R, R])
    initialize!(model)
    ids = bubble_ids(model)
    @test length(ids) == 2
    @test ids[1] != ids[2]
    LatticeBoltzmann.step!(model)
    @test bubble_ids(model) == ids
end

@testset "liquid tag is cleared and headspace is atmosphere" begin
    model = Model(8, 8, 8, 0.1; backend=CPU())
    domain = model.domains[1]
    fill!(domain.flags.data, TYPE_F)
    initialize!(model)
    domain.tag.data[1] = Int32(4)
    LatticeBoltzmann.step!(model)
    @test Array(domain.tag.data)[1] == 0
    @test bubble_count(model) == 0
    @test _no_surface_transition(model.foam.flags)

    N = 12
    model = Model(N, N, N, 0.1; backend=CPU(), σ=0)
    domain = model.domains[1]
    fill!(domain.flags.data, TYPE_F)
    for z in 8:(N - 1), y in 0:(N - 1), x in 0:(N - 1)
        n = x + y * N + z * N * N + 1
        domain.flags.data[n] = TYPE_G
        domain.ϕ.data[n] = 0
    end
    initialize!(model)
    LatticeBoltzmann.step!(model)
    tags = Array(domain.tag.data)
    flags = Array(domain.flags.data)
    @test bubble_count(model) == 0
    ngas = 0
    for i in eachindex(flags)
        if (flags[i] & TYPE_SU) == TYPE_G
            ngas += 1
            @test tags[i] == Int32(-1)
        end
        @test tags[i] <= 0
    end
    @test ngas > 0
end

@testset "failed punch rolls the table back" begin
    N = 16
    model = Model(N, N, N, 0.1; backend=CPU())
    domain = model.domains[1]
    fill!(domain.flags.data, TYPE_F)
    solid = 2 + 2 * N + 2 * N * N + 1
    domain.flags.data[solid] = TYPE_S
    @test_throws ArgumentError nucleate_bubbles!(
        model, [(8.5, 8.5, 8.5), (2.5, 2.5, 2.5)], [2.5, 2.0])
    @test bubble_count(model) == 0
    flags = Array(domain.flags.data)
    @test flags[solid] == TYPE_S
    @test !any(i -> (flags[i] & TYPE_SU) == TYPE_G, eachindex(flags))
    @test !any(i -> (flags[i] & TYPE_SU) == TYPE_I, eachindex(flags))
end

@testset "shell-only nucleus keeps its id" begin
    N = 12
    R = 0.4
    model = Model(N, N, N, 0.1; backend=CPU(), σ=0)
    domain = model.domains[1]
    fill!(domain.flags.data, TYPE_F)
    nucleate_bubbles!(model, [(6.5, 6.5, 6.5)], [R])
    @test bubble_count(model) == 1
    id = only(bubble_ids(model))
    flags = Array(domain.flags.data)
    ϕ = Array(domain.ϕ.data)
    @test !any(i -> (flags[i] & TYPE_SU) == TYPE_G, eachindex(flags))
    center = 6 + 6 * N + 6 * N * N + 1
    @test (flags[center] & TYPE_SU) == TYPE_I
    @test 0 < ϕ[center] < 1
    initialize!(model)
    LatticeBoltzmann.step!(model)
    @test bubble_ids(model) == [id]
end

@testset "TYPE_F box has zero bubbles" begin
    model = Model(8, 8, 8, 0.1; backend=CPU())
    domain = model.domains[1]
    fill!(domain.flags.data, TYPE_F)
    initialize!(model)
    LatticeBoltzmann.step!(model)
    LatticeBoltzmann.step!(model)
    @test bubble_count(model) == 0
    @test all(iszero, Array(domain.tag.data))
    @test _no_surface_transition(model.foam.flags)
end
