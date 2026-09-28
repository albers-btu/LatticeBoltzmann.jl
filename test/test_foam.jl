using Test, LatticeBoltzmann
using KernelAbstractions: CPU, synchronize

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
    row = model.foam.bubbles[Int(id)]
    imposed = Float32(row.ratio * (row.V_ref / row.V))
    ρb = Array(domain.ρb.data)
    tags = Array(domain.tag.data)
    @test any(>(0), tags)
    @test all(i -> tags[i] > 0 ? ρb[i] == imposed : ρb[i] == one(eltype(ρb)), eachindex(ρb))
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

@testset "imposed gas density" begin
    N = 11
    ϕ = zeros(Float32, N * N * N)
    R = 3.0f0
    mid = (N - 1) / 2
    for z in 0:(N - 1), y in 0:(N - 1), x in 0:(N - 1)
        d = sqrt((x - mid)^2 + (y - mid)^2 + (z - mid)^2)
        ϕ[x + y * N + z * N * N + 1] = clamp(d - R + 0.5f0, 0f0, 1f0)
    end
    n = findfirst(v -> 0f0 < v < 1f0, ϕ)
    @test n !== nothing
    n0 = n - 1
    x = n0 % N
    y = (n0 ÷ N) % N
    z = n0 ÷ (N * N)
    ϕ0 = ϕ[n]
    σ = 0.05f0
    phij = LatticeBoltzmann.gather_phi_d3q27(ϕ, ϕ0, x, y, z, N, N, N)
    κ = calculate_curvature(phij)
    @test abs(κ) > 1f-3
    expect = clamp(1f0 - 6f0 * σ * κ, 0.2f0, 2f0)
    @test LatticeBoltzmann.gas_density_plic(σ, ϕ, ϕ0, x, y, z, N, N, N, 1f0, 0f0) == expect
    @test LatticeBoltzmann.gas_density_plic(σ, ϕ, ϕ0, x, y, z, N, N, N) == expect
    @test LatticeBoltzmann.gas_density_plic(0f0, ϕ, ϕ0, x, y, z, N, N, N, 1.2f0, 0f0) == 1.2f0
    @test LatticeBoltzmann.gas_density_plic(0f0, ϕ, ϕ0, x, y, z, N, N, N, 5f0, 0f0) == 2f0
    @test LatticeBoltzmann.gas_density_plic(0f0, ϕ, ϕ0, x, y, z, N, N, N, 0f0, 0f0) == 0.2f0
end

@testset "doubling volume halves imposed density" begin
    foam = LatticeBoltzmann.FoamHost{Float32}(4)
    ratio = 1.5
    V = 8.0
    push!(foam.bubbles, LatticeBoltzmann.Bubble(V, V, ratio, false))
    foam.tag[1] = Int32(1)
    foam.tag[2] = Int32(-1)
    foam.tag[3] = Int32(0)
    foam.tag[4] = Int32(1)
    LatticeBoltzmann._impose_bubble_density!(foam, 1.0f0)
    @test foam.ρb[1] == Float32(ratio)
    @test foam.ρb[4] == Float32(ratio)
    @test foam.ρb[2] == 1f0
    @test foam.ρb[3] == 1f0
    foam.bubbles[1] = LatticeBoltzmann.Bubble(2V, V, ratio, false)
    LatticeBoltzmann._impose_bubble_density!(foam, 1.0f0)
    @test foam.ρb[1] == Float32(ratio / 2)
    @test foam.ρb[4] == Float32(ratio / 2)
    @test foam.bubbles[1].ratio == ratio
    @test foam.ρb[2] == 1f0
    @test foam.ρb[3] == 1f0
end

function _write_rest_feq!(fi, Nx, Ny, Nz, ρ::Float32)
    N = Nx * Ny * Nz
    w = weights(:D3Q19, Float32)
    c = velocities(:D3Q19)
    z = 0f0
    uu = 0f0
    for zz in 0:(Nz - 1), yy in 0:(Ny - 1), xx in 0:(Nx - 1)
        n = xx + yy * Nx + zz * Nx * Ny + 1
        fi[LatticeBoltzmann.f_index(n, 1, N)] = LatticeBoltzmann.feq(w[1], ρ, z, z, z, uu, c[1], Float32)
        for k in 1:((length(c) - 1) ÷ 2)
            i = 2k
            src = LatticeBoltzmann.src_index(xx, yy, zz, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
            fp = LatticeBoltzmann.feq(w[i], ρ, z, z, z, uu, c[i], Float32)
            fm = LatticeBoltzmann.feq(w[i + 1], ρ, z, z, z, uu, c[i + 1], Float32)
            fi[LatticeBoltzmann.f_index(src, i, N)] = fp
            fi[LatticeBoltzmann.f_index(n, i + 1, N)] = fm
        end
    end
    return nothing
end

function _flat_gas_volume(ϕ, tag, id)
    V = 0.0
    for i in eachindex(ϕ)
        tag[i] == id || continue
        ϕn = Float64(ϕ[i])
        ϕn = ϕn < 0 ? 0.0 : (ϕn > 1 ? 1.0 : ϕn)
        V += 1 - ϕn
    end
    return V
end

@testset "flat bubble reconstructs at the imposed density" begin
    Nx = Ny = Nz = 8
    model = Model(Nx, Ny, Nz, 0.1; backend=CPU(), σ=0)
    set_foam!(model; γ_b=1)
    domain = model.domains[1]
    fill!(domain.flags.data, TYPE_F)
    nucleate_bubbles!(model, [(4.0, 4.0, 4.0)], [1.5])
    id = only(bubble_ids(model))
    empty!(model.foam.blockers)

    flags = fill(TYPE_F, Nx * Ny * Nz)
    ϕ = ones(Float32, Nx * Ny * Nz)
    tag = zeros(Int32, Nx * Ny * Nz)
    for z in 0:(Nz - 1), y in 0:(Ny - 1), x in 0:(Nx - 1)
        n = x + y * Nx + z * Nx * Ny + 1
        if z == 2 || z == 5
            flags[n] = TYPE_I
            ϕ[n] = 0.5f0
            tag[n] = id
        elseif z == 3 || z == 4
            flags[n] = TYPE_G
            ϕ[n] = 0f0
            tag[n] = id
        end
    end
    copyto!(domain.flags.data, flags)
    copyto!(domain.ϕ.data, ϕ)
    copyto!(domain.tag.data, tag)
    V = _flat_gas_volume(ϕ, tag, id)
    model.foam.bubbles[Int(id)] = LatticeBoltzmann.Bubble(V, V, 1.5, false)

    initialize!(model)
    ϕ1 = Array(domain.ϕ.data)
    tag1 = Array(domain.tag.data)
    V = _flat_gas_volume(ϕ1, tag1, id)
    model.foam.bubbles[Int(id)] = LatticeBoltzmann.Bubble(V, V, 1.5, false)
    LatticeBoltzmann.foam_host!(model, domain)
    row = model.foam.bubbles[Int(id)]
    @test row.ratio == 1.5
    @test row.V_ref == V
    @test row.V == V
    imposed = Float32(row.ratio * (row.V_ref / row.V))
    @test imposed == 1.5f0

    ρb = Array(domain.ρb.data)
    tags = Array(domain.tag.data)
    flags1 = Array(domain.flags.data)
    nI = nothing
    srcG = nothing
    for z in 0:(Nz - 1), y in 0:(Ny - 1), x in 0:(Nx - 1)
        n = x + y * Nx + z * Nx * Ny + 1
        (flags1[n] & TYPE_SU) == TYPE_I || continue
        src = LatticeBoltzmann.src_index(x, y, z, 0, 0, 1, Nx, Ny, Nz)
        (flags1[src] & TYPE_SU) == TYPE_G || continue
        nI = n
        srcG = src
        break
    end
    @test nI !== nothing
    @test tags[nI] == id
    @test ρb[nI] == 1.5f0
    @test all(i -> tags[i] > 0 ? ρb[i] == 1.5f0 : ρb[i] == 1f0, eachindex(ρb))

    N = Nx * Ny * Nz
    _write_rest_feq!(domain.fi.data, Nx, Ny, Nz, 1f0)
    i = 6 # +z
    fp_out = domain.fi.data[LatticeBoltzmann.f_index(srcG, i + 1, N)]
    w = model.weights
    c = model.velocities
    ρG = 1.5f0
    z = 0f0
    uu = 0f0
    fegp = LatticeBoltzmann.feq(w[i], ρG, z, z, z, uu, c[i], Float32)
    fegm = LatticeBoltzmann.feq(w[i + 1], ρG, z, z, z, uu, c[i + 1], Float32)
    expected = fegp - fp_out + fegm

    Nd = Int(domain.N)
    model.cached_surface_0_even_kernel!(
        domain.fi.data, domain.ρ.data, domain.u.data, domain.flags.data,
        domain.mass.data, domain.massex.data, domain.ϕ.data, domain.T.data,
        domain.fs.data, domain.gi.data,
        model.weights, model.velocities,
        domain.fx, domain.fy, domain.fz, domain.σ, domain.σT, domain.Tσ,
        domain.Λ_v, domain.T_v, domain.p0v, domain.β_v,
        Nd, Nx, Ny, Nz, domain.Eacc.data,
        domain.h.data, domain.Q.data, domain.ω_T,
        domain.ci.data, domain.ρb.data, domain.Pi.data, domain.k_H;
        ndrange = N)
    synchronize(model.backend)
    got = domain.fi.data[LatticeBoltzmann.f_index(srcG, i, N)]
    @test got == expected
end

function _layer_index(x, y, z, Nx, Ny)
    return x + y * Nx + z * Nx * Ny + 1
end

# One z-slab per entry: flag, tag, ϕ. x and y are uniform, so the Parker–Youngs
# normal of a flat cut is ±z and three crossings is a center-to-center gap of 3.
function _paint_zlayers!(domain, layers)
    Nx = Int(domain.Nx)
    Ny = Int(domain.Ny)
    Nz = Int(domain.Nz)
    length(layers) == Nz || error("expected $Nz layers")
    N = Nx * Ny * Nz
    flags = fill(TYPE_F, N)
    ϕ = ones(Float32, N)
    tag = zeros(Int32, N)
    for z in 0:(Nz - 1)
        fl, tg, ϕz = layers[z + 1]
        for y in 0:(Ny - 1), x in 0:(Nx - 1)
            n = _layer_index(x, y, z, Nx, Ny)
            flags[n] = fl
            tag[n] = Int32(tg)
            ϕ[n] = ϕz
        end
    end
    copyto!(domain.flags.data, flags)
    copyto!(domain.ϕ.data, ϕ)
    copyto!(domain.tag.data, tag)
    fill!(domain.Pi.data, 0)
    return flags, ϕ, tag
end

function _launch_disjoining!(model)
    domain = model.domains[1]
    Nx = Int(domain.Nx)
    Ny = Int(domain.Ny)
    Nz = Int(domain.Nz)
    model.cached_disjoining_kernel!(
        domain.ϕ.data, domain.flags.data, domain.tag.data, domain.Pi.data,
        domain.k_Π, Nx, Ny, Nz; ndrange = Nx * Ny * Nz)
    synchronize(model.backend)
    return Array(domain.Pi.data)
end

function _film_layers(Nz, z1, z2, tag1, tag2; film_tag=0)
    layers = [(TYPE_F, 0, 1.0f0) for _ in 1:Nz]
    # Gas on the outer face of each interface. The Parker–Youngs stencil is ±1.
    z1 > 0 && (layers[z1] = (TYPE_G, tag1, 0.0f0))
    z2 < Nz - 1 && (layers[z2 + 2] = (TYPE_G, tag2, 0.0f0))
    layers[z1 + 1] = (TYPE_I, tag1, 0.5f0)
    layers[z2 + 1] = (TYPE_I, tag2, 0.5f0)
    for z in (z1 + 1):(z2 - 1)
        layers[z + 1] = (TYPE_F, film_tag, 1.0f0)
    end
    return layers
end

function _expect_pi(ϕ, kΠ, n, j, x, y, z, Nx, Ny, Nz)
    nϕ = calculate_normal_py(LatticeBoltzmann.gather_phi_d3q27(ϕ, ϕ[n], x, y, z, Nx, Ny, Nz))
    δs = abs(plic_cube(ϕ[n], nϕ))
    δo = abs(plic_cube(ϕ[j], nϕ))
    # Axis-aligned march: three crossings, each tDelta = 1.
    d = max(3.0f0 - δs - δo, 0.0f0)
    return kΠ * (4.0f0 - d), nϕ, d
end

@testset "disjoining across a liquid film" begin
    Nx, Ny, Nz = 4, 4, 16
    kΠ = 0.05f0
    model = Model(Nx, Ny, Nz, 0.1; backend=CPU(), σ=0)
    set_foam!(model; k_Π=kΠ)
    domain = model.domains[1]
    # Interfaces at z=2 and z=5: two TYPE_F, tag-0 cells between them.
    layers = _film_layers(Nz, 2, 5, 1, 2)
    _paint_zlayers!(domain, layers)
    fill!(domain.ρ.data, 1.3f0)
    fill!(domain.ρb.data, 1.0f0)
    ϕ_before = Array(domain.ϕ.data)
    ρ_before = Array(domain.ρ.data)
    Pi = _launch_disjoining!(model)
    @test Array(domain.ϕ.data) == ϕ_before
    @test Array(domain.ρ.data) == ρ_before

    n_if = 0
    for z in (2, 5), y in 0:(Ny - 1), x in 0:(Nx - 1)
        n = _layer_index(x, y, z, Nx, Ny)
        zhit = z == 2 ? 5 : 2
        j = _layer_index(x, y, zhit, Nx, Ny)
        expect, nϕ, d = _expect_pi(ϕ_before, kΠ, n, j, x, y, z, Nx, Ny, Nz)
        @test abs(nϕ[1]) < 1f-5 && abs(nϕ[2]) < 1f-5
        @test abs(nϕ[3]) > 0.99f0
        @test 0 <= d < 4
        @test Pi[n] ≈ expect atol=1f-4
        @test Pi[n] > 0
        ρg = LatticeBoltzmann.gas_density_plic(0f0, ϕ_before, ϕ_before[n], x, y, z, Nx, Ny, Nz, 1f0, Pi[n])
        ρg0 = LatticeBoltzmann.gas_density_plic(0f0, ϕ_before, ϕ_before[n], x, y, z, Nx, Ny, Nz, 1f0, 0f0)
        @test ρg0 == 1f0
        @test ρg ≈ ρg0 - 3f0 * Pi[n] atol=1f-5
        n_if += 1
    end
    @test n_if == 2 * Nx * Ny
    for z in 0:(Nz - 1)
        z == 2 && continue
        z == 5 && continue
        @test all(iszero, (Pi[_layer_index(x, y, z, Nx, Ny)] for y in 0:(Ny - 1) for x in 0:(Nx - 1)))
    end

    # Same id on both sides: the other interface is skipped, not a hit.
    _paint_zlayers!(domain, _film_layers(Nz, 2, 5, 1, 1))
    @test all(iszero, _launch_disjoining!(model))

    # Five liquid cells: the other interface is past s = 4.
    _paint_zlayers!(domain, _film_layers(Nz, 2, 8, 1, 2))
    @test all(iszero, _launch_disjoining!(model))

    # Atmosphere in the film ends the walk before the other bubble.
    layers_atm = _film_layers(Nz, 2, 5, 1, 2; film_tag=-1)
    _paint_zlayers!(domain, layers_atm)
    @test all(iszero, _launch_disjoining!(model))

    # Constant fill: Parker–Youngs normal is zero.
    N = Nx * Ny * Nz
    flags = fill(TYPE_F, N)
    ϕ = fill(0.5f0, N)
    tag = zeros(Int32, N)
    n0 = _layer_index(1, 1, 4, Nx, Ny)
    flags[n0] = TYPE_I
    tag[n0] = Int32(1)
    copyto!(domain.flags.data, flags)
    copyto!(domain.ϕ.data, ϕ)
    copyto!(domain.tag.data, tag)
    fill!(domain.Pi.data, 0)
    Pi0 = _launch_disjoining!(model)
    @test Pi0[n0] == 0
    @test all(iszero, Pi0)

    # k_Π = 0 skips the launch. Live rows keep the two ids, so a launch would write Π.
    set_foam!(model; k_Π=0)
    initialize!(model)
    _paint_zlayers!(domain, _film_layers(Nz, 2, 5, 1, 2))
    push!(model.foam.bubbles, LatticeBoltzmann.Bubble(4.0, 4.0, 1.0, false))
    push!(model.foam.bubbles, LatticeBoltzmann.Bubble(4.0, 4.0, 1.0, false))
    LatticeBoltzmann.step!(model)
    @test all(iszero, Array(domain.Pi.data))

    @test LatticeBoltzmann.gas_density_plic(0f0, ϕ, 0.5f0, 1, 1, 4, Nx, Ny, Nz, 1f0, 0.1f0) == 0.7f0
end

@testset "stale disjoining pressure is cleared" begin
    Nx, Ny, Nz = 4, 4, 16
    model = Model(Nx, Ny, Nz, 0.1; backend=CPU(), σ=0)
    set_foam!(model; k_Π=0.05)
    domain = model.domains[1]
    initialize!(model)
    _paint_zlayers!(domain, _film_layers(Nz, 2, 5, 1, 2))
    n = _layer_index(1, 1, 2, Nx, Ny)
    Pi = _launch_disjoining!(model)
    @test Pi[n] > 0
    flags = Array(domain.flags.data)
    tags = Array(domain.tag.data)
    flags[n] = TYPE_F
    tags[n] = Int32(0)
    copyto!(domain.flags.data, flags)
    copyto!(domain.tag.data, tags)
    @test Array(domain.Pi.data)[n] > 0
    LatticeBoltzmann.step!(model)
    @test iszero(Array(domain.Pi.data)[n])

    fill!(domain.Pi.data, 0.4f0)
    set_foam!(model; k_Π=0)
    @test all(iszero, Array(domain.Pi.data))

    stale = zeros(Float32, length(domain.Pi))
    stale[n] = 0.4f0
    copyto!(domain.Pi.data, stale)
    LatticeBoltzmann.step!(model)
    @test all(iszero, Array(domain.Pi.data))
end

# Sum of the seven populations as the next collide would load them.
# Completed steps leave domain.t = nsteps; the next substep uses isodd(t).
function _foam_macro_c(domain, n, x, y, z, t_odd::Val{odd}) where {odd}
    N = domain.N
    Nx = Int(domain.Nx); Ny = Int(domain.Ny); Nz = Int(domain.Nz)
    ci = domain.ci.data
    CType = eltype(ci)
    s = CType(ci[LatticeBoltzmann.f_index(n, 1, N)])
    for k in 1:3
        i = 2k
        cx = k == 1 ? 1 : 0
        cy = k == 2 ? 1 : 0
        cz = k == 3 ? 1 : 0
        src = LatticeBoltzmann.src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
        fp, fm = LatticeBoltzmann.load_pair(ci, n, src, i, t_odd, N, CType)
        s += fp + fm
    end
    return s
end

function foam_macro_c(domain, n, x, y, z)
    # n_hydro is 1, so the next collide parity is isodd(domain.t).
    if isodd(Int(domain.t))
        return _foam_macro_c(domain, n, x, y, z, Val(true))
    else
        return _foam_macro_c(domain, n, x, y, z, Val(false))
    end
end

function foam_mean_c(domain)
    return sum(Float64, Array(domain.ci.data)) / domain.N
end

function foam_sine_amplitude(domain)
    Nx = Int(domain.Nx); Ny = Int(domain.Ny); Nz = Int(domain.Nz)
    acc = 0.0
    norm = 0.0
    for z in 0:Nz-1, y in 0:Ny-1, x in 0:Nx-1
        n = 1 + x + Nx * (y + Ny * z)
        s = sin(2π * x / Nx)
        acc += Float64(foam_macro_c(domain, n, x, y, z)) * s
        norm += s * s
    end
    return acc / norm
end

@testset "pure diffusion of a sine" begin
    Nx, Ny, Nz = 48, 4, 4
    D = 0.05
    model = Model(Nx, Ny, Nz, 0.1; backend=CPU())
    domain = model.domains[1]
    fill!(domain.flags.data, TYPE_F)
    fill!(domain.T.data, 1.25f0)
    set_foam!(model; D=D, q=0, c0=0)
    initialize!(model)
    N = domain.N
    for z in 0:Nz-1, y in 0:Ny-1, x in 0:Nx-1
        n = 1 + x + Nx * (y + Ny * z)
        cv = Float32(sin(2π * x / Nx))
        LatticeBoltzmann.store_ceq!(domain.ci.data, n, x, y, z, cv,
                                    0.0f0, 0.0f0, 0.0f0, N, Nx, Ny, Nz, Val(false))
    end
    gi0 = copy(Array(domain.gi.data))
    @test !all(iszero, gi0)
    @test foam_sine_amplitude(domain) ≈ 1.0 atol=1.0e-5

    for _ in 1:200
        LatticeBoltzmann.step!(model)
    end
    @test Int(domain.t) == 200
    @test all(==(TYPE_F), Array(domain.flags.data))
    k = 2π / Nx
    A = exp(-D * k * k * 200)
    @test foam_sine_amplitude(domain) ≈ A rtol=0.03
    @test Array(domain.gi.data) == gi0
    @test maximum(abs, Array(domain.u.data)) < 1.0f-5
end

@testset "uniform concentration is stationary" begin
    c0 = 0.25f0
    model = Model(8, 8, 8, 0.1; backend=CPU())
    domain = model.domains[1]
    fill!(domain.flags.data, TYPE_F)
    fill!(domain.T.data, 1.25f0)
    set_foam!(model; D=0.05, q=0, c0=c0)
    initialize!(model)
    gi0 = copy(Array(domain.gi.data))
    @test !all(iszero, gi0)
    m0 = foam_mean_c(domain)
    @test m0 ≈ Float64(c0) atol=1.0e-5

    LatticeBoltzmann.step!(model) # even
    m1 = foam_mean_c(domain)
    @test m1 ≈ Float64(c0) atol=1.0e-5

    LatticeBoltzmann.step!(model) # odd
    m2 = foam_mean_c(domain)
    @test m2 ≈ Float64(c0) atol=1.0e-5
    @test abs(m2 - m0) < 1.0e-5
    @test Array(domain.gi.data) == gi0
end

@testset "source q on liquid only" begin
    c0 = 0.2f0
    q = 0.015f0

    model = Model(8, 8, 8, 0.1; backend=CPU())
    domain = model.domains[1]
    fill!(domain.flags.data, TYPE_F)
    set_foam!(model; D=0.05, q=q, c0=c0)
    initialize!(model)
    m0 = foam_mean_c(domain)
    LatticeBoltzmann.step!(model)
    @test foam_mean_c(domain) ≈ m0 + Float64(q) atol=1.0e-5
    LatticeBoltzmann.step!(model)
    @test foam_mean_c(domain) ≈ m0 + 2 * Float64(q) atol=1.0e-5

    modelI = Model(8, 8, 8, 0.1; backend=CPU())
    domainI = modelI.domains[1]
    fill!(domainI.flags.data, TYPE_I)
    set_foam!(modelI; D=0.05, q=q, c0=c0)
    initialize!(modelI)
    mi0 = foam_mean_c(domainI)
    LatticeBoltzmann.step!(modelI)
    @test foam_mean_c(domainI) ≈ mi0 atol=1.0e-5
end
