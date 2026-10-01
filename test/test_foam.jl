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
    @test Array(domain.ϕ_old.data) == Array(domain.ϕ.data)

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
    raw = 1f0 - 6f0 * σ * κ
    @test LatticeBoltzmann.gas_density_plic(σ, ϕ, ϕ0, x, y, z, N, N, N, 1f0, 0f0) == clamp(raw, 0.8f0, 1.6f0)
    @test LatticeBoltzmann.gas_density_plic(σ, ϕ, ϕ0, x, y, z, N, N, N) == clamp(raw, 0.2f0, 2f0)
    @test LatticeBoltzmann.gas_density_plic(0f0, ϕ, ϕ0, x, y, z, N, N, N, 1.2f0, 0f0) == 1.2f0
    @test LatticeBoltzmann.gas_density_plic(0f0, ϕ, ϕ0, x, y, z, N, N, N, 5f0, 0f0) == 1.6f0
    @test LatticeBoltzmann.gas_density_plic(0f0, ϕ, ϕ0, x, y, z, N, N, N, 0f0, 0f0) == 0.8f0
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
        domain.ci.data, domain.flux.data, domain.ρb.data, domain.Pi.data, domain.k_H, domain.D,
        domain.tag.data, domain.c.data;
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
        @test ρg ≈ clamp(ρg0 - 3f0 * Pi[n], 0.8f0, 1.6f0) atol=1f-5
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

    # Three liquid cells. The ray reaches the other interface on the step
    # where s = 4. Half-fill has zero PLIC offset, so d = 4 and Π stays 0.
    # A fill of 0.25 leaves d < 4 after both offsets (paper eq. 30).
    layers3 = _film_layers(Nz, 2, 6, 1, 2)
    layers3[3] = (TYPE_I, 1, 0.25f0)
    layers3[7] = (TYPE_I, 2, 0.25f0)
    _paint_zlayers!(domain, layers3)
    Pi3 = _launch_disjoining!(model)
    ϕ3 = Array(domain.ϕ.data)
    n3 = _layer_index(1, 1, 2, Nx, Ny)
    j3 = _layer_index(1, 1, 6, Nx, Ny)
    nϕ3 = calculate_normal_py(LatticeBoltzmann.gather_phi_d3q27(ϕ3, ϕ3[n3], 1, 1, 2, Nx, Ny, Nz))
    d3 = max(4.0f0 - abs(plic_cube(ϕ3[n3], nϕ3)) - abs(plic_cube(ϕ3[j3], nϕ3)), 0.0f0)
    @test d3 < 4
    @test Pi3[n3] ≈ kΠ * (4.0f0 - d3) atol=1f-4
    @test Pi3[n3] > 0

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

    @test LatticeBoltzmann.gas_density_plic(0f0, ϕ, 0.5f0, 1, 1, 4, Nx, Ny, Nz, 1f0, 0.05f0) == 0.85f0
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

function _foam_max_c_err(domain, c0)
    Nx = Int(domain.Nx); Ny = Int(domain.Ny); Nz = Int(domain.Nz)
    err = 0.0
    for z in 0:(Nz - 1), y in 0:(Ny - 1), x in 0:(Nx - 1)
        n = 1 + x + Nx * (y + Ny * z)
        err = max(err, abs(Float64(foam_macro_c(domain, n, x, y, z)) - Float64(c0)))
    end
    return err
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
    @test _foam_max_c_err(domain, c0) < 1.0e-5

    LatticeBoltzmann.step!(model) # odd
    m2 = foam_mean_c(domain)
    @test m2 ≈ Float64(c0) atol=1.0e-5
    @test abs(m2 - m0) < 1.0e-5
    @test _foam_max_c_err(domain, c0) < 1.0e-5
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

function _launch_surface0_even!(model)
    domain = model.domains[1]
    Nx = Int(domain.Nx)
    Ny = Int(domain.Ny)
    Nz = Int(domain.Nz)
    model.cached_surface_0_even_kernel!(
        domain.fi.data, domain.ρ.data, domain.u.data, domain.flags.data,
        domain.mass.data, domain.massex.data, domain.ϕ.data, domain.T.data,
        domain.fs.data, domain.gi.data,
        model.weights, model.velocities,
        domain.fx, domain.fy, domain.fz, domain.σ, domain.σT, domain.Tσ,
        domain.Λ_v, domain.T_v, domain.p0v, domain.β_v,
        Int(domain.N), Nx, Ny, Nz, domain.Eacc.data,
        domain.h.data, domain.Q.data, domain.ω_T,
        domain.ci.data, domain.flux.data, domain.ρb.data, domain.Pi.data, domain.k_H, domain.D,
        domain.tag.data, domain.c.data;
        ndrange = Nx * Ny * Nz)
    synchronize(model.backend)
    return nothing
end

@testset "Henry reconstructs gas-facing links at k_H ρ_b / 3" begin
    Nx = Ny = Nz = 8
    model = Model(Nx, Ny, Nz, 0.1; backend=CPU(), σ=0)
    domain = model.domains[1]
    N = Nx * Ny * Nz
    fill!(domain.flags.data, TYPE_F)
    set_foam!(model; D=0.03, k_H=0.001, c0=0)
    initialize!(model)

    x, y, z = 3, 3, 3
    n = _layer_index(x, y, z, Nx, Ny)
    src_g = LatticeBoltzmann.src_index(x, y, z, 1, 0, 0, Nx, Ny, Nz)
    flags = fill(TYPE_F, N)
    flags[n] = TYPE_I
    flags[src_g] = TYPE_G
    copyto!(domain.flags.data, flags)
    ϕ = ones(Float32, N)
    ϕ[n] = 0.5f0
    ϕ[src_g] = 0f0
    copyto!(domain.ϕ.data, ϕ)
    fill!(domain.ρb.data, 1f0)
    fill!(domain.Pi.data, 0f0)
    fill!(domain.fs.data, 0f0)
    fill!(domain.u.data, 0f0)
    _write_rest_feq!(domain.fi.data, Nx, Ny, Nz, 1f0)

    k_H = 0.001f0
    c_H = k_H * (1f0 / 3f0)
    ceq = LatticeBoltzmann.ceq_axis(c_H, 0f0)
    ceq4 = LatticeBoltzmann.ceq_axis(k_H * (1f0 / 4f0), 0f0)
    @test ceq != ceq4
    i = 2
    out_slot = LatticeBoltzmann.f_index(src_g, i + 1, N)
    gas_slot = LatticeBoltzmann.f_index(src_g, i, N)
    sentinel = 0.17f0

    function load_ci!(; outgoing=ceq)
        fill!(domain.ci.data, sentinel)
        domain.ci.data[out_slot] = outgoing
        return copy(Array(domain.ci.data))
    end

    before = load_ci!()
    gi0 = copy(Array(domain.gi.data))
    _launch_surface0_even!(model)
    ci = Array(domain.ci.data)
    @test ci[gas_slot] == ceq
    @test ci[gas_slot] != ceq4
    ci[gas_slot] = before[gas_slot]
    @test ci == before
    @test Array(domain.gi.data) == gi0

    # Solidified interface keeps ρ_gas = 1 and does not write ci.
    domain.fs.data[n] = 1f0
    before = load_ci!()
    _launch_surface0_even!(model)
    @test Array(domain.ci.data) == before
    @test LatticeBoltzmann.is_solid_fraction(1f0)

    domain.fs.data[n] = 0f0
    set_foam!(model; D=0, k_H=0.001)
    before = load_ci!()
    _launch_surface0_even!(model)
    @test Array(domain.ci.data) == before

    set_foam!(model; D=0.03, k_H=0)
    before = load_ci!()
    _launch_surface0_even!(model)
    @test Array(domain.ci.data) == before
end

function _launch_ϕ_correction!(model)
    domain = model.domains[1]
    Nx = Int(domain.Nx)
    Ny = Int(domain.Ny)
    Nz = Int(domain.Nz)
    model.cached_ϕ_correction_kernel!(
        domain.ϕ.data, domain.ϕ_old.data, domain.c.data,
        domain.flags.data, domain.tag.data, domain.flux.data,
        model.velocities, Nx, Ny, Nz; ndrange = Nx * Ny * Nz)
    synchronize(model.backend)
    return Array(domain.flux.data)
end

@testset "fill change bins onto a face or an edge neighbor" begin
    Nx = Ny = Nz = 8
    model = Model(Nx, Ny, Nz, 0.1; backend=CPU(), σ=0)
    domain = model.domains[1]
    x, y, z = 3, 3, 3
    n = _layer_index(x, y, z, Nx, Ny)
    face = LatticeBoltzmann.src_index(x, y, z, 1, 0, 0, Nx, Ny, Nz)
    edge = LatticeBoltzmann.src_index(x, y, z, 1, 1, 0, Nx, Ny, Nz)
    @test face != edge

    function paint!(tagged)
        N = Nx * Ny * Nz
        flags = fill(TYPE_F, N)
        flags[n] = TYPE_I
        ϕ = ones(Float32, N)
        ϕ_old = ones(Float32, N)
        cfield = zeros(Float32, N)
        tag = zeros(Int32, N)
        ϕ[n] = 0.5f0
        ϕ_old[n] = 1f0
        cfield[n] = 0.4f0
        tag[tagged] = Int32(1)
        copyto!(domain.flags.data, flags)
        copyto!(domain.ϕ.data, ϕ)
        copyto!(domain.ϕ_old.data, ϕ_old)
        copyto!(domain.c.data, cfield)
        copyto!(domain.tag.data, tag)
        fill!(domain.flux.data, 0)
        return nothing
    end

    paint!(face)
    flux = _launch_ϕ_correction!(model)
    @test flux[1] == 0.2f0
    @test all(iszero, @view flux[2:end])

    paint!(edge)
    flux = _launch_ϕ_correction!(model)
    @test flux[1] == 0.2f0
    @test all(iszero, @view flux[2:end])
end

@testset "dissolved gas grows one bubble" begin
    N = 32
    R0 = 3.0
    Δc = 0.5
    D = 0.03
    k_H = 0.001
    V_m = 3.0
    model = Model(N, N, N, 0.25; backend=CPU(), σ=0, n_hydro=1)
    domain = model.domains[1]
    flags = fill(TYPE_F, N * N * N)
    for z in 0:(N - 1), y in 0:(N - 1), x in 0:(N - 1)
        if x == 0 || y == 0 || z == 0 || x == N - 1 || y == N - 1 || z == N - 1
            flags[1 + x + N * y + N * N * z] = TYPE_S
        end
    end
    copyto!(domain.flags.data, flags)
    nucleate_bubbles!(model, [(N / 2, N / 2, N / 2)], [R0])
    set_foam!(model; D=D, k_H=k_H, k_Π=0, q=0, V_m=V_m, γ_b=1,
              c0=k_H / 3 + Δc, ρ_liquid=1)
    initialize!(model)
    id = only(bubble_ids(model))
    R_i = (3 * bubble_volume(model, id) / (4π))^(1 / 3)
    for _ in 1:200
        LatticeBoltzmann.step!(model)
    end
    R_f = (3 * bubble_volume(model, id) / (4π))^(1 / 3)
    @test R_f > R_i
    @test bubble_ids(model) == [id]
end

function _blank_foam(Nx, Ny, Nz)
    foam = LatticeBoltzmann.FoamHost{Float32}(Nx * Ny * Nz)
    fill!(foam.flags, TYPE_F)
    fill!(foam.ϕ, one(Float32))
    return foam
end

function _paint_gas!(foam, Nx, Ny, cells, id)
    for (x, y, z) in cells
        n = x + y * Nx + z * Nx * Ny + 1
        foam.flags[n] = TYPE_G
        foam.ϕ[n] = 0
        foam.tag_prev[n] = Int32(id)
    end
    return nothing
end

function _put_row!(foam, id, V, V_ref, ratio; frozen=false)
    while length(foam.bubbles) < id
        push!(foam.bubbles, nothing)
    end
    foam.bubbles[id] = LatticeBoltzmann.Bubble(V, V_ref, ratio, frozen)
    return nothing
end

function _live_rows(foam)
    out = Tuple{Int,LatticeBoltzmann.Bubble}[]
    for (i, row) in enumerate(foam.bubbles)
        row === nothing && continue
        push!(out, (i, row))
    end
    return out
end

function _liquid_tags_are_zero(foam)
    for i in eachindex(foam.tag)
        (foam.flags[i] & TYPE_SU) == TYPE_F || continue
        foam.tag[i] == 0 || return false
    end
    return true
end

# Host retag only. Ids 5 and 2 touch along x; the survivor is 2, not a new id.
@testset "merge keeps min_id_not_upstream" begin
    Nx, Ny, Nz = 10, 4, 4
    y = z = 1
    foam = _blank_foam(Nx, Ny, Nz)
    _paint_gas!(foam, Nx, Ny, [(2, y, z), (3, y, z)], 5)
    _paint_gas!(foam, Nx, Ny, [(4, y, z), (5, y, z)], 2)
    _put_row!(foam, 5, 1.0, 4.0, 1.0)
    _put_row!(foam, 2, 1.0, 4.0, 3.0)
    before = 1.0 * 4.0 + 3.0 * 4.0
    LatticeBoltzmann._retag!(foam, Nx, Ny, Nz)
    rows = _live_rows(foam)
    @test length(rows) == 1
    id, row = only(rows)
    @test id == 2
    @test row.ratio == (1.0 * 4.0 + 3.0 * 4.0) / (4.0 + 4.0)
    @test row.V_ref == 8.0
    @test row.ratio * row.V_ref == before
    @test foam.bubbles[5] === nothing
    @test _liquid_tags_are_zero(foam)
    for x in (2, 3, 4, 5)
        @test foam.tag[x + y * Nx + z * Nx * Ny + 1] == Int32(2)
    end
end

@testset "split copies ratio and rescales V_ref" begin
    Nx, Ny, Nz = 10, 4, 4
    y = z = 1
    foam = _blank_foam(Nx, Ny, Nz)
    # One cell, a liquid wall, then two cells. Parent V_ref is not Σ V.
    _paint_gas!(foam, Nx, Ny, [(2, y, z)], 3)
    _paint_gas!(foam, Nx, Ny, [(4, y, z), (5, y, z)], 3)
    _put_row!(foam, 3, 9.0, 6.0, 2.0)
    LatticeBoltzmann._retag!(foam, Nx, Ny, Nz)
    rows = _live_rows(foam)
    @test length(rows) == 2
    @test any(r -> r[1] == 3, rows)
    @test all(r -> r[2].ratio == 2.0, rows)
    @test sum(r -> r[2].V_ref, rows) == 6.0
    @test sum(r -> r[2].ratio * r[2].V_ref, rows) == 2.0 * 6.0
    small = only(r for r in rows if r[2].V == 1.0)
    large = only(r for r in rows if r[2].V == 2.0)
    @test small[2].V_ref == 6.0 * 1.0 / 3.0
    @test large[2].V_ref == 6.0 * 2.0 / 3.0
    @test _liquid_tags_are_zero(foam)
end

@testset "erased bubble drops its row" begin
    Nx, Ny, Nz = 10, 4, 4
    y = z = 1
    foam = _blank_foam(Nx, Ny, Nz)
    keep = [(2, y, z), (3, y, z)]
    gone = [(6, y, z), (7, y, z)]
    _paint_gas!(foam, Nx, Ny, keep, 1)
    _paint_gas!(foam, Nx, Ny, gone, 2)
    for (x, yy, zz) in gone
        n = x + yy * Nx + zz * Nx * Ny + 1
        foam.flags[n] = TYPE_F
        foam.ϕ[n] = 1
    end
    _put_row!(foam, 1, 2.0, 5.0, 1.5)
    _put_row!(foam, 2, 2.0, 5.0, 9.0)
    LatticeBoltzmann._retag!(foam, Nx, Ny, Nz)
    rows = _live_rows(foam)
    @test length(rows) == 1
    id, row = only(rows)
    @test id == 1
    @test row.ratio == 1.5
    @test row.V_ref == 5.0
    @test foam.bubbles[2] === nothing
    @test _liquid_tags_are_zero(foam)
    for (x, yy, zz) in gone
        @test foam.tag[x + yy * Nx + zz * Nx * Ny + 1] == 0
    end
end

@testset "bubble joined to atmosphere is dropped" begin
    Nx, Ny, Nz = 12, 4, 4
    y = z = 1
    foam = _blank_foam(Nx, Ny, Nz)
    joined = [(2, y, z), (3, y, z)]
    atm = [(4, y, z), (5, y, z)]
    spectator = [(8, y, z), (9, y, z)]
    _paint_gas!(foam, Nx, Ny, joined, 1)
    _paint_gas!(foam, Nx, Ny, atm, -1)
    _paint_gas!(foam, Nx, Ny, spectator, 2)
    _put_row!(foam, 1, 2.0, 4.0, 1.5)
    _put_row!(foam, 2, 2.0, 7.0, 2.5)
    LatticeBoltzmann._retag!(foam, Nx, Ny, Nz)
    rows = _live_rows(foam)
    @test length(rows) == 1
    id, row = only(rows)
    @test id == 2
    @test row.ratio == 2.5
    @test row.V_ref == 7.0
    @test foam.bubbles[1] === nothing
    for (x, yy, zz) in vcat(joined, atm)
        @test foam.tag[x + yy * Nx + zz * Nx * Ny + 1] == Int32(-1)
    end
    for (x, yy, zz) in spectator
        @test foam.tag[x + yy * Nx + zz * Nx * Ny + 1] == Int32(2)
    end
    @test _liquid_tags_are_zero(foam)
end

@testset "untagged gas between a pore and the atmosphere is plugged" begin
    Nx, Ny, Nz = 12, 4, 4
    y = z = 1
    foam = _blank_foam(Nx, Ny, Nz)
    pore = [(2, y, z), (3, y, z)]
    bridge = [(4, y, z)]
    atm = [(5, y, z), (6, y, z)]
    spectator = [(8, y, z), (9, y, z)]
    _paint_gas!(foam, Nx, Ny, pore, 1)
    _paint_gas!(foam, Nx, Ny, bridge, 0)
    _paint_gas!(foam, Nx, Ny, atm, -1)
    _paint_gas!(foam, Nx, Ny, spectator, 2)
    _put_row!(foam, 1, 2.0, 4.0, 1.5)
    _put_row!(foam, 2, 2.0, 7.0, 2.5)
    plugs = LatticeBoltzmann._seal_atmosphere!(foam, Nx, Ny, Nz)
    @test length(plugs) == 1
    n3 = 3 + y * Nx + z * Nx * Ny + 1
    @test (foam.flags[n3] & TYPE_SU) == TYPE_F
    @test foam.ϕ[n3] == 1
    LatticeBoltzmann._retag!(foam, Nx, Ny, Nz)
    @test foam.bubbles[1] !== nothing
    @test foam.bubbles[1].ratio == 1.5
    @test foam.bubbles[1].V_ref == 4.0
    @test foam.bubbles[1].V == 1.0
    @test foam.bubbles[2] !== nothing
    @test foam.bubbles[2].ratio == 2.5
    @test foam.bubbles[2].V == 2.0
    @test foam.tag[2 + y * Nx + z * Nx * Ny + 1] == Int32(1)
    for (x, yy, zz) in vcat(bridge, atm)
        @test foam.tag[x + yy * Nx + zz * Nx * Ny + 1] == Int32(-1)
    end
    for (x, yy, zz) in spectator
        @test foam.tag[x + yy * Nx + zz * Nx * Ny + 1] == Int32(2)
    end
end

@testset "untagged gas that does not reach the atmosphere stays with the pore" begin
    Nx, Ny, Nz = 10, 4, 4
    y = z = 1
    foam = _blank_foam(Nx, Ny, Nz)
    _paint_gas!(foam, Nx, Ny, [(2, y, z), (3, y, z)], 1)
    _paint_gas!(foam, Nx, Ny, [(4, y, z)], 0)
    _put_row!(foam, 1, 2.0, 4.0, 1.5)
    @test isempty(LatticeBoltzmann._seal_atmosphere!(foam, Nx, Ny, Nz))
    @test (foam.flags[3 + y * Nx + z * Nx * Ny + 1] & TYPE_SU) == TYPE_G
end

@testset "frozen bubble touching atmosphere is kept" begin
    Nx, Ny, Nz = 12, 4, 4
    y = z = 1
    foam = _blank_foam(Nx, Ny, Nz)
    joined = [(2, y, z), (3, y, z)]
    atm = [(4, y, z), (5, y, z)]
    spectator = [(8, y, z), (9, y, z)]
    _paint_gas!(foam, Nx, Ny, joined, 1)
    _paint_gas!(foam, Nx, Ny, atm, -1)
    _paint_gas!(foam, Nx, Ny, spectator, 2)
    _put_row!(foam, 1, 2.0, 4.0, 1.5; frozen=true)
    _put_row!(foam, 2, 2.0, 7.0, 2.5)
    LatticeBoltzmann._retag!(foam, Nx, Ny, Nz)
    @test foam.bubbles[1] !== nothing
    @test foam.bubbles[1].frozen
    @test foam.bubbles[1].ratio == 1.5
    @test foam.bubbles[1].V_ref == 4.0
    @test foam.bubbles[1].V == 2.0
    id, row = only(r for r in _live_rows(foam) if r[1] == 2)
    @test row.ratio == 2.5
    @test row.V_ref == 7.0
    for (x, yy, zz) in joined
        @test foam.tag[x + yy * Nx + zz * Nx * Ny + 1] == Int32(1)
    end
    for (x, yy, zz) in atm
        @test foam.tag[x + yy * Nx + zz * Nx * Ny + 1] == Int32(-1)
    end
    for (x, yy, zz) in spectator
        @test foam.tag[x + yy * Nx + zz * Nx * Ny + 1] == Int32(2)
    end
end

@testset "frozen bubbles that share a face do not merge" begin
    Nx, Ny, Nz = 10, 4, 4
    y = z = 1
    foam = _blank_foam(Nx, Ny, Nz)
    _paint_gas!(foam, Nx, Ny, [(2, y, z), (3, y, z)], 1)
    _paint_gas!(foam, Nx, Ny, [(4, y, z), (5, y, z)], 2)
    _put_row!(foam, 1, 2.0, 4.0, 1.5; frozen=true)
    _put_row!(foam, 2, 2.0, 5.0, 3.0; frozen=true)
    LatticeBoltzmann._retag!(foam, Nx, Ny, Nz)
    rows = _live_rows(foam)
    @test length(rows) == 2
    @test foam.bubbles[1].ratio == 1.5
    @test foam.bubbles[1].V_ref == 4.0
    @test foam.bubbles[1].frozen
    @test foam.bubbles[2].ratio == 3.0
    @test foam.bubbles[2].V_ref == 5.0
    @test foam.bubbles[2].frozen
    for x in (2, 3)
        @test foam.tag[x + y * Nx + z * Nx * Ny + 1] == Int32(1)
    end
    for x in (4, 5)
        @test foam.tag[x + y * Nx + z * Nx * Ny + 1] == Int32(2)
    end
end

@testset "frozen bubble stays while a hot neighbor vents" begin
    Nx, Ny, Nz = 12, 4, 4
    y = z = 1
    foam = _blank_foam(Nx, Ny, Nz)
    _paint_gas!(foam, Nx, Ny, [(2, y, z), (3, y, z)], 1)
    _paint_gas!(foam, Nx, Ny, [(4, y, z), (5, y, z)], 2)
    _paint_gas!(foam, Nx, Ny, [(6, y, z), (7, y, z)], -1)
    _put_row!(foam, 1, 2.0, 4.0, 1.25; frozen=true)
    _put_row!(foam, 2, 2.0, 4.0, 2.0)
    LatticeBoltzmann._retag!(foam, Nx, Ny, Nz)
    @test foam.bubbles[1] !== nothing
    @test foam.bubbles[1].frozen
    @test foam.bubbles[1].ratio == 1.25
    @test foam.bubbles[1].V_ref == 4.0
    @test foam.bubbles[2] === nothing
    for x in (2, 3)
        @test foam.tag[x + y * Nx + z * Nx * Ny + 1] == Int32(1)
    end
    for x in (4, 5, 6, 7)
        @test foam.tag[x + y * Nx + z * Nx * Ny + 1] == Int32(-1)
    end
end

@testset "frozen row ignores dissolved flux" begin
    foam = _blank_foam(4, 4, 4)
    _put_row!(foam, 1, 2.0, 4.0, 1.5; frozen=true)
    _put_row!(foam, 2, 2.0, 4.0, 1.5)
    flux = zeros(Float32, 4)
    flux[1] = 1.0f0
    flux[2] = 1.0f0
    LatticeBoltzmann._add_dissolved_inventory!(foam, flux, 2.0, 1.0)
    @test foam.bubbles[1].ratio == 1.5
    @test foam.bubbles[1].frozen
    @test foam.bubbles[2].ratio == 1.5 + 1.0 * 2.0 / 4.0
    @test !foam.bubbles[2].frozen
end

function _paint_device_gas!(domain, Nx, Ny, cells_ids)
    N = length(domain.flags)
    flags = fill(TYPE_F, N)
    ϕ = fill(1.0f0, N)
    tag = zeros(Int32, N)
    y = z = 1
    for (x, id) in cells_ids
        n = x + y * Nx + z * Nx * Ny + 1
        flags[n] = TYPE_G
        ϕ[n] = 0
        tag[n] = Int32(id)
    end
    copyto!(domain.flags.data, flags)
    copyto!(domain.ϕ.data, ϕ)
    copyto!(domain.tag.data, tag)
    return nothing
end

@testset "merge weights flux from both parents" begin
    Nx, Ny, Nz = 10, 4, 4
    model = Model(Nx, Ny, Nz, 0.1; backend=CPU())
    domain = model.domains[1]
    set_foam!(model; V_m=2, ρ_liquid=1)
    _paint_device_gas!(domain, Nx, Ny, [(2, 5), (3, 5), (4, 2), (5, 2)])
    _put_row!(model.foam, 5, 1.0, 4.0, 1.0)
    _put_row!(model.foam, 2, 1.0, 4.0, 3.0)
    flux = Array(domain.flux.data)
    flux[5] = 0.5f0
    flux[2] = 1.0f0
    copyto!(domain.flux.data, flux)
    scale = 2.0
    r5 = 1.0 + 0.5 * scale / 4.0
    r2 = 3.0 + 1.0 * scale / 4.0
    LatticeBoltzmann.foam_host!(model, domain)
    rows = _live_rows(model.foam)
    @test length(rows) == 1
    id, row = only(rows)
    @test id == 2
    @test row.V_ref == 8.0
    @test row.ratio == (r5 * 4.0 + r2 * 4.0) / 8.0
    @test row.ratio * row.V_ref == 1.0 * 4.0 + 3.0 * 4.0 + scale * (0.5 + 1.0)
    @test all(iszero, Array(domain.flux.data))
end

@testset "split shares the flux-updated ratio" begin
    Nx, Ny, Nz = 10, 4, 4
    model = Model(Nx, Ny, Nz, 0.1; backend=CPU())
    domain = model.domains[1]
    set_foam!(model; V_m=2, ρ_liquid=1)
    _paint_device_gas!(domain, Nx, Ny, [(2, 3), (4, 3), (5, 3)])
    _put_row!(model.foam, 3, 9.0, 6.0, 2.0)
    flux = Array(domain.flux.data)
    flux[3] = 1.5f0
    copyto!(domain.flux.data, flux)
    scale = 2.0
    r = 2.0 + 1.5 * scale / 6.0
    LatticeBoltzmann.foam_host!(model, domain)
    rows = _live_rows(model.foam)
    @test length(rows) == 2
    @test any(row -> row[1] == 3, rows)
    @test all(row -> row[2].ratio == r, rows)
    @test sum(row -> row[2].V_ref, rows) == 6.0
    @test sum(row -> row[2].ratio * row[2].V_ref, rows) == r * 6.0
end
function _vtk_find(hay::Vector{UInt8}, needle::Vector{UInt8})
    n = length(needle)
    last = length(hay) - n + 1
    last < 1 && return nothing
    for i in 1:last
        if hay[i] == needle[1] && @view(hay[i:(i + n - 1)]) == needle
            return i
        end
    end
    return nothing
end

function _vtk_attr(tag, key)
    m = match(Regex("\\b$(key)=\"([^\"]*)\""), tag)
    return m === nothing ? nothing : m.captures[1]
end

function _zlib_uncompress(src::Vector{UInt8}, dst_len::Int)
    dst = Vector{UInt8}(undef, dst_len)
    destLen = Ref{Culong}(Culong(dst_len))
    ret = ccall((:uncompress, "libz"), Cint,
        (Ptr{UInt8}, Ptr{Culong}, Ptr{UInt8}, Culong),
        dst, destLen, src, Culong(length(src)))
    ret == 0 || error("zlib uncompress failed ($ret)")
    Int(destLen[]) == dst_len || error("zlib uncompress wrote $(destLen[]) bytes, expected $dst_len")
    return dst
end

# Appended VTK XML from export!. Returns the DataArray type and the point values.
function _vtk_point_data(path, name, nvals::Int)
    bytes = read(path)
    needle = Vector{UInt8}("<AppendedData encoding=\"raw\">\n_")
    at = _vtk_find(bytes, needle)
    at === nothing && error("no appended VTK data in $path")
    header = String(@view(bytes[1:(at - 1)]))
    m = match(Regex("<DataArray\\b[^>]*\\bName=\"$(name)\"[^>]*>"), header)
    m === nothing && error("VTK file has no DataArray $name")
    tag = m.match
    typ = _vtk_attr(tag, "type")
    off = parse(Int, _vtk_attr(tag, "offset"))
    nc = parse(Int, something(_vtk_attr(tag, "NumberOfComponents"), "1"))
    T = typ == "Int32" ? Int32 : typ == "Float32" ? Float32 : error("VTK $name has type $typ")
    ht = occursin("header_type=\"UInt32\"", header) ? UInt32 : UInt64
    compressed = occursin("compressor=\"vtkZLibDataCompressor\"", header)
    io = IOBuffer(bytes)
    seek(io, at + length(needle) - 1 + off)
    if compressed
        nblocks = Int(read(io, ht))
        1 <= nblocks <= 128 || error("unexpected VTK block count $nblocks")
        blocksize = Int(read(io, ht))
        last_blocksize = Int(read(io, ht))
        sizes = [Int(read(io, ht)) for _ in 1:nblocks]
        raw = UInt8[]
        for (k, sz) in enumerate(sizes)
            append!(raw, _zlib_uncompress(read(io, sz), k == nblocks ? last_blocksize : blocksize))
        end
    else
        raw = read(io, Int(read(io, ht)))
    end
    n = nvals * nc
    length(raw) == n * sizeof(T) || error("VTK $name has $(length(raw)) bytes, expected $(n * sizeof(T))")
    vals = Vector{T}(undef, n)
    read!(IOBuffer(raw), vals)
    return typ, vals
end

@testset "vtk exports c and tag" begin
    fields = (:rho, :p, :u, :phi, :flags)
    offs, ncomp = LatticeBoltzmann._vtk_offsets(LatticeBoltzmann._vtk_mask(fields))
    @test ncomp == 7
    @test (offs.rho, offs.p, offs.u, offs.T, offs.fs, offs.phi, offs.mp, offs.S, offs.Q, offs.flags) ==
        (0, 1, 2, -1, -1, 5, -1, -1, -1, 6)
    @test offs.c == -1 && offs.tag == -1
    both, nboth = LatticeBoltzmann._vtk_offsets(LatticeBoltzmann._vtk_mask((fields..., :c, :tag)))
    @test nboth == 9
    @test both.rho == 0 && both.flags == 6 && both.c == 7 && both.tag == 8

    N = 16
    model = Model(N, N, N, 0.1; backend=CPU(), σ=0)
    domain = model.domains[1]
    fill!(domain.flags.data, TYPE_F)
    set_foam!(model; D=0, c0=0.25f0)
    nucleate_bubbles!(model, [(8.5, 8.5, 8.5)], [3.5])
    id = only(bubble_ids(model))
    mktempdir() do dir
        export!(model; dir, fields=(:c, :tag), sync=true)
        paths = filter(p -> endswith(p, ".vti") || endswith(p, ".vtr"), readdir(dir; join=true))
        @test length(paths) == 1
        ctyp, cvals = _vtk_point_data(paths[1], "c", domain.N)
        ttyp, tvals = _vtk_point_data(paths[1], "tag", domain.N)
        @test ctyp == "Float32"
        @test ttyp == "Int32"
        @test cvals == Array(domain.c.data)
        @test tvals == Array(domain.tag.data)
        @test any(==(0.25f0), cvals)
        @test any(iszero, cvals)
        flags = Array(domain.flags.data)
        n_gas = 0
        n_liquid = 0
        n_shell = 0
        for i in eachindex(flags)
            su = flags[i] & TYPE_SU
            if su == TYPE_G
                n_gas += 1
                @test tvals[i] == id
            elseif su == TYPE_I
                n_shell += 1
                @test tvals[i] == id
            elseif su == TYPE_F
                n_liquid += 1
                @test tvals[i] == Int32(0)
            end
        end
        @test n_gas > 0
        @test n_shell > 0
        @test n_liquid > 0
    end
end

function _shell_case(fs_shell)
    Nx, Ny, Nz = 8, 4, 4
    model = Model(Nx, Ny, Nz, 0.2; backend=CPU(), n_hydro=1)
    domain = model.domains[1]
    set_foam!(model; V_m=2, ρ_liquid=1)
    N = Nx * Ny * Nz
    flags = fill(TYPE_F, N)
    ϕ = fill(1.0f0, N)
    tag = zeros(Int32, N)
    y = z = 1
    # Gas at x = 3, shells at x = 2 and x = 4, both facing the gas.
    for (x, su, ph) in ((3, TYPE_G, 0.0f0), (2, TYPE_I, 0.4f0), (4, TYPE_I, 0.6f0))
        n = x + y * Nx + z * Nx * Ny + 1
        flags[n] = su
        ϕ[n] = ph
        tag[n] = Int32(1)
    end
    copyto!(domain.flags.data, flags)
    copyto!(domain.ϕ.data, ϕ)
    copyto!(domain.tag.data, tag)
    fs = zeros(Float32, N)
    fs[2 + y * Nx + z * Nx * Ny + 1] = fs_shell[1]
    fs[4 + y * Nx + z * Nx * Ny + 1] = fs_shell[2]
    copyto!(domain.fs.data, fs)
    _put_row!(model.foam, 1, 1.0, 4.0, 1.5)
    flux = Array(domain.flux.data)
    flux[1] = 1.0f0
    copyto!(domain.flux.data, flux)
    return model, domain
end

@testset "solid shell freezes and ignores flux" begin
    model, domain = _shell_case((1.0f0, 1.0f0))
    LatticeBoltzmann.foam_host!(model, domain)
    row = model.foam.bubbles[1]
    @test row !== nothing
    @test row.frozen
    @test row.ratio == 1.5
    @test row.V_ref == 4.0
    @test bubble_frozen(model, 1)
    tags = Array(domain.tag.data)
    y = z = 1
    Nx, Ny = 8, 4
    for x in (2, 3, 4)
        @test tags[x + y * Nx + z * Nx * Ny + 1] == Int32(1)
    end
    @test all(iszero, Array(domain.flux.data))
end

@testset "one liquid interface cell does not freeze" begin
    model, domain = _shell_case((1.0f0, 0.0f0))
    LatticeBoltzmann.foam_host!(model, domain)
    row = model.foam.bubbles[1]
    @test row !== nothing
    @test !row.frozen
    @test row.ratio == 1.5 + 1.0 * 2.0 / 4.0
    @test row.V_ref == 4.0
end

@testset "fs = 0 does not freeze" begin
    model, domain = _shell_case((0.0f0, 0.0f0))
    LatticeBoltzmann.foam_host!(model, domain)
    row = model.foam.bubbles[1]
    @test !row.frozen
    @test row.ratio == 1.5 + 0.5
end

@testset "a frozen bubble does not remelt" begin
    model, domain = _shell_case((0.0f0, 0.0f0))
    model.foam.bubbles[1] = LatticeBoltzmann.Bubble(1.0, 4.0, 1.5, true)
    LatticeBoltzmann.foam_host!(model, domain)
    row = model.foam.bubbles[1]
    @test row.frozen
    @test row.ratio == 1.5
    @test row.V_ref == 4.0
    @test bubble_ids(model) == Int32[1]
end

@testset "gas with no shell does not freeze" begin
    Nx, Ny, Nz = 6, 4, 4
    model = Model(Nx, Ny, Nz, 0.2; backend=CPU(), n_hydro=1)
    domain = model.domains[1]
    set_foam!(model; V_m=2, ρ_liquid=1)
    _paint_device_gas!(domain, Nx, Ny, [(2, 1), (3, 1)])
    fill!(domain.fs.data, 1.0f0)
    _put_row!(model.foam, 1, 2.0, 4.0, 1.5)
    flux = Array(domain.flux.data)
    flux[1] = 0.5f0
    copyto!(domain.flux.data, flux)
    LatticeBoltzmann.foam_host!(model, domain)
    row = model.foam.bubbles[1]
    @test row !== nothing
    @test !row.frozen
    @test row.ratio == 1.5 + 0.5 * 2.0 / 4.0
end

# Cold mold, one bubble, enthalpy. The shell has to solidify from the walls
# and then hold its id, ratio, and volume.
@testset "cooling mold freezes one bubble" begin
    N = 14
    R0 = 2.5
    model = Model(N, N, N, 0.2; backend=CPU(), σ=0.02, n_hydro=1,
                  α=0.4, Λ=0.2, Ts=0.7, Tl=0.78, K0=1.0f-3, T_avg=1.0, fz=-1.0f-4)
    domain = model.domains[1]
    flags = fill(TYPE_F, N * N * N)
    Th = fill(1.0f0, N * N * N)
    fsh = zeros(Float32, N * N * N)
    for z in 0:(N - 1), y in 0:(N - 1), x in 0:(N - 1)
        n = 1 + x + N * y + N * N * z
        if x == 0 || y == 0 || z == 0 || x == N - 1 || y == N - 1 || z == N - 1
            flags[n] = TYPE_S | TYPE_T
            Th[n] = 0.45f0
            fsh[n] = 1.0f0
        end
    end
    copyto!(domain.flags.data, flags)
    copyto!(domain.T.data, Th)
    copyto!(domain.fs.data, fsh)
    nucleate_bubbles!(model, [(N / 2, N / 2, N / 2)], [R0])
    set_foam!(model; D=0.02, k_H=1.0f-4, k_Π=0, q=0, V_m=3, γ_b=1,
              c0=0.05, ρ_liquid=1)
    initialize!(model)
    id0 = only(bubble_ids(model))
    frozen_at = 0
    for t in 1:300
        LatticeBoltzmann.step!(model)
        ids = bubble_ids(model)
        id0 in ids || break
        if !isempty(ids) && all(i -> bubble_frozen(model, i), ids)
            frozen_at = t
            break
        end
    end
    @test frozen_at > 0
    ids_f = bubble_ids(model)
    @test id0 in ids_f
    @test all(i -> bubble_frozen(model, i), ids_f)
    ratios = Dict(i => bubble_ratio(model, i) for i in ids_f)
    vols = Dict(i => bubble_volume(model, i) for i in ids_f)
    @test vols[id0] > 0.4 * (4π / 3) * R0^3
    for _ in 1:40
        LatticeBoltzmann.step!(model)
    end
    ids_h = bubble_ids(model)
    @test ids_h == ids_f
    for i in ids_h
        @test bubble_frozen(model, i)
        @test bubble_ratio(model, i) == ratios[i]
        @test bubble_volume(model, i) ≈ vols[i] rtol=0.02
    end
    flags = Array(domain.flags.data)
    fs = Array(domain.fs.data)
    tags = Array(domain.tag.data)
    n_shell = 0
    n_solid = 0
    for i in eachindex(flags)
        (flags[i] & TYPE_SU) == TYPE_I || continue
        tags[i] in ids_h || continue
        n_shell += 1
        LatticeBoltzmann.is_solid_fraction(fs[i]) && (n_solid += 1)
    end
    @test n_shell > 0
    @test n_solid == n_shell
end

function _metal_T_range(domain)
    flags = Array(domain.flags.data)
    T = Array(domain.T.data)
    tmin = Inf
    tmax = -Inf
    s = 0.0
    n = 0
    twall = Inf
    for i in eachindex(flags)
        if (flags[i] & TYPE_T) != 0x00
            twall = min(twall, Float64(T[i]))
        end
        su = flags[i] & TYPE_SU
        su == TYPE_F || su == TYPE_I || continue
        t = Float64(T[i])
        tmin = min(tmin, t)
        tmax = max(tmax, t)
        s += t
        n += 1
    end
    return tmin, tmax, n == 0 ? 0.0 : s / n, twall, n
end

# Gas cells skip the thermal collide. Without a zero-flux reconstruction the
# stale populations are a heat sink and the metal falls through the mold
# temperature. Λ = 0 keeps the shell liquid, so this is the pore link itself
# and not the freeze. σ = 0 and g = 0 so the pore is not driven into a wall.
@testset "insulating gas does not cool the melt below the mold" begin
    N = 14
    R0 = 2.5
    Tmold = 0.40f0
    Tinit = 1.20f0
    model = Model(N, N, N, 0.2; backend=CPU(), σ=0.0, n_hydro=1,
                  α=0.4, Λ=0.0, Ts=0.7, Tl=0.8, K0=0.0, T_avg=1.0, fz=0)
    domain = model.domains[1]
    flags = fill(TYPE_F, N * N * N)
    Th = fill(Tinit, N * N * N)
    fsh = zeros(Float32, N * N * N)
    for z in 0:(N - 1), y in 0:(N - 1), x in 0:(N - 1)
        n = 1 + x + N * y + N * N * z
        if x == 0 || y == 0 || z == 0 || x == N - 1 || y == N - 1 || z == N - 1
            flags[n] = TYPE_S | TYPE_T
            Th[n] = Tmold
            fsh[n] = 1
        end
    end
    copyto!(domain.flags.data, flags)
    copyto!(domain.T.data, Th)
    copyto!(domain.fs.data, fsh)
    nucleate_bubbles!(model, [(N / 2, N / 2, N / 2)], [R0])
    set_foam!(model; D=0, k_H=0, k_Π=0, q=0, V_m=0, γ_b=1, c0=0, ρ_liquid=1)
    initialize!(model)
    id0 = only(bubble_ids(model))
    V0 = bubble_volume(model, id0)
    for _ in 1:200
        LatticeBoltzmann.step!(model)
    end
    tmin, tmax, tmean, twall, nmet = _metal_T_range(domain)
    @info "insulating gas" tmin tmax tmean twall nmet V=bubble_volume(model, id0)
    @test nmet > 0
    @test twall ≈ Float64(Tmold) atol=1e-5
    @test tmin >= Float64(Tmold) - 0.04
    @test tmean < Float64(Tinit) - 0.25
    @test tmax < 0.9
    @test bubble_ids(model) == Int32[id0]
    @test bubble_volume(model, id0) > 0.5 * V0
    @test all(isfinite, Array(domain.T.data))
end

# Open mold, two pores, gravity, a little dissolved gas. They have to cool,
# freeze, and still be the same two bubbles. Venting into the headspace or
# coalescing drops an id.
@testset "open mold freezes two bubbles in place" begin
    Nx, Ny, Nz = 18, 18, 26
    H = 16
    R0 = 2.2
    Tmold = 0.45f0
    Tinit = 1.05f0
    model = Model(Nx, Ny, Nz, 0.2; backend=CPU(), σ=0.02, n_hydro=1,
                  α=0.35, Λ=0.2, Ts=0.70, Tl=0.80, K0=1.0f-3, T_avg=1.0, fz=-3.0f-5)
    domain = model.domains[1]
    flags = fill(TYPE_G, Nx * Ny * Nz)
    Th = fill(Tinit, Nx * Ny * Nz)
    fsh = zeros(Float32, Nx * Ny * Nz)
    for z in 0:(Nz - 1), y in 0:(Ny - 1), x in 0:(Nx - 1)
        n = 1 + x + Nx * y + Nx * Ny * z
        if x == 0 || y == 0 || z == 0 || x == Nx - 1 || y == Ny - 1 || z == Nz - 1
            flags[n] = TYPE_S | TYPE_T
            Th[n] = Tmold
            fsh[n] = 1
        elseif z < H
            flags[n] = TYPE_F
        end
    end
    copyto!(domain.flags.data, flags)
    copyto!(domain.T.data, Th)
    copyto!(domain.fs.data, fsh)
    # Far enough apart, and far enough under the free surface, that the
    # pair does not pinch while it cools. The free surface is not a Henry sink.
    centers = [(5.0, 5.0, 6.0), (13.0, 13.0, 6.0)]
    nucleate_bubbles!(model, centers, [R0, R0])
    set_foam!(model; D=0.005, k_H=1.0f-4, k_Π=0.01, q=0, V_m=3, γ_b=1,
              c0=0.02, ρ_liquid=1)
    initialize!(model)
    ids0 = bubble_ids(model)
    @test length(ids0) == 2
    V0 = Dict(i => bubble_volume(model, i) for i in ids0)
    frozen_at = 0
    for t in 1:400
        LatticeBoltzmann.step!(model)
        ids = bubble_ids(model)
        # A pinch of a couple of cells gets its own id and then fills back in.
        # A real second pore, or the loss of one of these two, is a failure.
        missing = !isempty(setdiff(ids0, ids))
        spawned = any(i -> !(i in ids0) && bubble_volume(model, i) >= 4, ids)
        (missing || spawned) && break
        if issetequal(ids, ids0) && all(i -> bubble_frozen(model, i), ids)
            frozen_at = t
            break
        end
    end
    tmin, tmax, tmean, twall, _ = _metal_T_range(domain)
    @info "open mold" frozen_at ids=bubble_ids(model) tmin tmax tmean twall
    @test frozen_at > 0
    @test issetequal(bubble_ids(model), ids0)
    @test all(i -> bubble_frozen(model, i), bubble_ids(model))
    @test tmin >= Float64(Tmold) - 0.04
    @test twall ≈ Float64(Tmold) atol=1e-5
    ratios = Dict(i => bubble_ratio(model, i) for i in ids0)
    vols = Dict(i => bubble_volume(model, i) for i in ids0)
    for i in ids0
        @test vols[i] > 0.5 * V0[i]
    end
    for _ in 1:40
        LatticeBoltzmann.step!(model)
    end
    @test issetequal(bubble_ids(model), ids0)
    for i in ids0
        @test bubble_frozen(model, i)
        @test bubble_ratio(model, i) == ratios[i]
        @test bubble_volume(model, i) ≈ vols[i] rtol=0.02
    end
    tmin2, _, _, _, _ = _metal_T_range(domain)
    @test tmin2 >= Float64(Tmold) - 0.04
    @test all(isfinite, Array(domain.T.data))
end

# Metal α with the mold permeability. drag = 2ρ stored a zero velocity and
# reflected the population momentum, so the melt temperature left the mold
# while the printed speed stayed ~0. A closed liquid box has no gas in it.
@testset "mush permeability does not drive temperature outside the mold" begin
    N = 14
    Nz = 16
    Tmold = 0.73f0
    Tinit = 1.15f0
    model = Model(N, N, Nz, 0.2; backend=CPU(), σ=0.0, n_hydro=1,
                  α=0.04, Λ=0.406, Ts=0.955, Tl=1.0, K0=3.0f-3, T_avg=1.0, fz=0)
    domain = model.domains[1]
    flags = fill(TYPE_F, N * N * Nz)
    Th = fill(Tinit, N * N * Nz)
    fsh = zeros(Float32, N * N * Nz)
    for z in 0:(Nz - 1), y in 0:(N - 1), x in 0:(N - 1)
        n = 1 + x + N * y + N * N * z
        if x == 0 || y == 0 || z == 0 || x == N - 1 || y == N - 1 || z == Nz - 1
            flags[n] = TYPE_S | TYPE_T
            Th[n] = Tmold
            fsh[n] = 1
        end
    end
    copyto!(domain.flags.data, flags)
    copyto!(domain.T.data, Th)
    copyto!(domain.fs.data, fsh)
    set_foam!(model; D=0, k_H=0, k_Π=0, q=0, V_m=0, γ_b=1, c0=0, ρ_liquid=1)
    initialize!(model)
    for _ in 1:160
        LatticeBoltzmann.step!(model)
    end
    tmin, tmax, tmean, twall, nmet = _metal_T_range(domain)
    @info "mush permeability" tmin tmax tmean twall nmet
    @test nmet > 0
    @test twall ≈ Float64(Tmold) atol=1e-5
    @test tmin >= Float64(Tmold) - 0.04
    @test tmax <= Float64(Tinit) + 0.05
    @test tmean < Float64(Tinit) - 0.1
    @test all(isfinite, Array(domain.T.data))
end

# Headspace fi used to stay 0. Reconstruction then did feq - 0 + feq, the
# free surface accelerated, and the melt under it heated above the pour.
@testset "open surface does not heat the melt above the pour" begin
    Nx, Ny, Nz = 16, 16, 22
    H = 14
    Tmold = 0.73f0
    Tinit = 1.15f0
    model = Model(Nx, Ny, Nz, 0.2; backend=CPU(), σ=0.02, n_hydro=1,
                  α=0.04, Λ=0.406, Ts=0.955, Tl=1.0, K0=3.0f-3, T_avg=Tinit, fz=-2.0f-5)
    domain = model.domains[1]
    flags = fill(TYPE_G, Nx * Ny * Nz)
    Th = fill(Tinit, length(flags))
    fsh = zeros(Float32, length(flags))
    for z in 0:(Nz - 1), y in 0:(Ny - 1), x in 0:(Nx - 1)
        n = 1 + x + Nx * y + Nx * Ny * z
        if x == 0 || y == 0 || z == 0 || x == Nx - 1 || y == Ny - 1 || z == Nz - 1
            flags[n] = TYPE_S | TYPE_T
            Th[n] = Tmold
            fsh[n] = 1
        elseif z < H
            flags[n] = TYPE_F
        end
    end
    copyto!(domain.flags.data, flags)
    copyto!(domain.T.data, Th)
    copyto!(domain.fs.data, fsh)
    set_foam!(model; D=0, k_H=0, k_Π=0, q=0, V_m=0, γ_b=1, c0=0, ρ_liquid=1)
    initialize!(model)
    for _ in 1:30
        LatticeBoltzmann.step!(model)
    end
    tmin, tmax, _, twall, nmet = _metal_T_range(domain)
    @info "open surface" tmin tmax twall nmet
    @test nmet > 0
    @test tmin >= Float64(Tmold) - 0.04
    @test tmax <= Float64(Tinit) + 0.05
    @test all(isfinite, Array(domain.T.data))
end

# The open top is not a pore. Henry at c_H ≈ 0 would empty the melt.
# The free-surface link is anti-bounce-back at the interface concentration.
@testset "free surface does not vent dissolved gas" begin
    Nx = Ny = Nz = 8
    model = Model(Nx, Ny, Nz, 0.1; backend=CPU(), σ=0)
    domain = model.domains[1]
    N = Nx * Ny * Nz
    fill!(domain.flags.data, TYPE_F)
    set_foam!(model; D=0.03, k_H=0.001, c0=0)
    initialize!(model)

    x, y, z = 3, 3, 3
    n = _layer_index(x, y, z, Nx, Ny)
    src_g = LatticeBoltzmann.src_index(x, y, z, 1, 0, 0, Nx, Ny, Nz)
    flags = fill(TYPE_F, N)
    flags[n] = TYPE_I
    flags[src_g] = TYPE_G
    copyto!(domain.flags.data, flags)
    ϕ = ones(Float32, N)
    ϕ[n] = 0.5f0
    ϕ[src_g] = 0f0
    copyto!(domain.ϕ.data, ϕ)
    tag = zeros(Int32, N)
    tag[n] = Int32(-1)
    tag[src_g] = Int32(-1)
    copyto!(domain.tag.data, tag)
    fill!(domain.ρb.data, 1f0)
    fill!(domain.Pi.data, 0f0)
    fill!(domain.fs.data, 0f0)
    fill!(domain.u.data, 0f0)
    _write_rest_feq!(domain.fi.data, Nx, Ny, Nz, 1f0)

    # Liquid neighbors carry the bath concentration. The interface cell's
    # own c is not the no-flux value.
    fill!(domain.c.data, 0.2f0)
    domain.c.data[src_g] = 0f0
    # Anti-BB writes 2 ceq(c_wall) - outgoing. A pore still uses
    # c_H = k_H ρ_b / 3, and that slot equals ceq only when outgoing is ceq.
    ceq_face = LatticeBoltzmann.ceq_axis(0.2f0, 0f0)
    ceq_vent = LatticeBoltzmann.ceq_axis(0.001f0 / 3f0, 0f0)
    i = 2
    out_slot = LatticeBoltzmann.f_index(src_g, i + 1, N)
    gas_slot = LatticeBoltzmann.f_index(src_g, i, N)
    sentinel = 0.17f0
    fill!(domain.ci.data, sentinel)
    domain.ci.data[out_slot] = ceq_vent
    _launch_surface0_even!(model)
    @test domain.ci.data[gas_slot] ≈ (2f0 * ceq_face - ceq_vent) atol=1f-5
    @test domain.ci.data[gas_slot] != sentinel

    # The same geometry with a pore tag still pins the gas link to c_H.
    domain.tag.data[n] = Int32(1)
    domain.tag.data[src_g] = Int32(1)
    fill!(domain.ci.data, sentinel)
    domain.ci.data[out_slot] = ceq_vent
    _launch_surface0_even!(model)
    @test domain.ci.data[gas_slot] ≈ ceq_vent atol=1f-6
    @test domain.ci.data[gas_slot] != sentinel
end

# One gas cell touching the headspace used to drop the whole pore.
# The bridge cell is turned back into liquid and the id survives.
@testset "gas bridge into the headspace does not drop the pore" begin
    Nx = Ny = 8
    Nz = 16
    H = 10
    model = Model(Nx, Ny, Nz, 0.1; backend=CPU(), σ=0, n_hydro=1)
    domain = model.domains[1]
    N = Nx * Ny * Nz
    flags = fill(TYPE_G, N)
    for z in 0:(Nz - 1), y in 0:(Ny - 1), x in 0:(Nx - 1)
        n = _layer_index(x, y, z, Nx, Ny)
        if x == 0 || y == 0 || z == 0 || x == Nx - 1 || y == Ny - 1 || z == Nz - 1
            flags[n] = TYPE_S
        elseif z < H
            flags[n] = TYPE_F
        end
    end
    copyto!(domain.flags.data, flags)
    set_foam!(model; D=0.01, k_H=1e-5, k_Π=0, V_m=3, c0=0.02)
    nucleate_bubbles!(model, [(4.0, 4.0, 5.0)], [2.2])
    initialize!(model)
    ids0 = bubble_ids(model)
    @test length(ids0) == 1
    id = ids0[1]
    V0 = bubble_volume(model, id)
    flags = Array(domain.flags.data)
    tags = Array(domain.tag.data)
    # The headspace is tag −1 only after a retag. Paint that before the bridge.
    for i in eachindex(flags)
        if (flags[i] & TYPE_SU) == TYPE_G && tags[i] == Int32(0)
            tags[i] = Int32(-1)
        end
    end
    x = 4
    y = 4
    top = -1
    az = Nz
    for z in 0:(Nz - 1)
        n = _layer_index(x, y, z, Nx, Ny)
        su = flags[n] & TYPE_SU
        if tags[n] == id && su == TYPE_G
            top = max(top, z)
        elseif tags[n] == Int32(-1) && su == TYPE_G
            az = min(az, z)
        end
    end
    @test top >= 0 && az > top + 1
    ϕ = Array(domain.ϕ.data)
    nbridge = 0
    for z in (top + 1):(az - 1)
        n = _layer_index(x, y, z, Nx, Ny)
        flags[n] = TYPE_G
        tags[n] = id
        ϕ[n] = 0
        nbridge += 1
    end
    @test nbridge >= 1
    copyto!(domain.flags.data, flags)
    copyto!(domain.tag.data, tags)
    copyto!(domain.ϕ.data, ϕ)
    for _ in 1:8
        LatticeBoltzmann.step!(model)
        @test id in bubble_ids(model)
    end
    @test bubble_volume(model, id) > 0.5 * V0
end

# Henry removes (old − reconstructed) from each gas-facing slot and stages
# that drop on flux. step! then parks it on the TYPE_F rests. Eq. 27 adds
# the liquid-side difference on top and removes it from the interface rest.
@testset "Henry drop is the pore credit" begin
    Nx = Ny = Nz = 8
    model = Model(Nx, Ny, Nz, 0.1; backend=CPU(), σ=0)
    domain = model.domains[1]
    N = Nx * Ny * Nz
    fill!(domain.flags.data, TYPE_F)
    set_foam!(model; D=0.03, k_H=0.001, c0=0)
    initialize!(model)
    x = y = z = 3
    n = _layer_index(x, y, z, Nx, Ny)
    src_g = LatticeBoltzmann.src_index(x, y, z, 1, 0, 0, Nx, Ny, Nz)
    i = 2
    flags = fill(TYPE_F, N)
    flags[n] = TYPE_I
    flags[src_g] = TYPE_G
    copyto!(domain.flags.data, flags)
    ϕ = ones(Float32, N)
    ϕ[n] = 0.5f0
    ϕ[src_g] = 0f0
    copyto!(domain.ϕ.data, ϕ)
    fill!(domain.ρb.data, 1f0)
    fill!(domain.Pi.data, 0f0)
    fill!(domain.fs.data, 0f0)
    fill!(domain.u.data, 0f0)
    fill!(domain.tag.data, Int32(0))
    domain.tag.data[n] = Int32(1)
    _write_rest_feq!(domain.fi.data, Nx, Ny, Nz, 1f0)

    ceq = LatticeBoltzmann.ceq_axis(0.001f0 / 3f0, 0f0)
    gas_slot = LatticeBoltzmann.f_index(src_g, i, N)
    out_slot = LatticeBoltzmann.f_index(src_g, i + 1, N)
    fill!(domain.ci.data, 0.2f0)
    # Liquid-face imbalance fp_in − fm_out on the −x neighbor.
    domain.ci.data[LatticeBoltzmann.f_index(n, i + 1, N)] = 0.3f0
    domain.ci.data[LatticeBoltzmann.f_index(n, i, N)] = 0.1f0
    old_gas = domain.ci.data[gas_slot]
    fp_out = domain.ci.data[out_slot]
    δ_slot = old_gas - (ceq - fp_out + ceq)
    sum0 = sum(Array(domain.ci.data))
    fill!(domain.flux.data, 0f0)
    _launch_surface0_even!(model)
    ci = Array(domain.ci.data)
    @test ci[gas_slot] ≈ (ceq - fp_out + ceq) atol=1f-5
    @test sum(ci) ≈ sum0 - δ_slot atol=1f-4
    @test Array(domain.flux.data)[1] ≈ δ_slot atol=1f-5
    @test all(iszero, @view Array(domain.flux.data)[2:end])
    own0 = ci[LatticeBoltzmann.f_index(n, 1, N)]

    # Eq. 27 is booked on top of the staged drop. It does not rewrite the
    # interface rest: that cell is the Dirichlet shell. step! parks the
    # staged drop on the bath before this kernel; this launch does not.
    model.cached_concentration_flux_even_kernel!(
        domain.ci.data, domain.flags.data, domain.tag.data, domain.flux.data,
        Int(domain.N), Nx, Ny, Nz; ndrange = N)
    synchronize(model.backend)
    @test Array(domain.flux.data)[1] ≈ δ_slot + 0.2f0 atol=1f-5
    @test Array(domain.ci.data)[LatticeBoltzmann.f_index(n, 1, N)] ≈ own0 atol=1f-5

    # No liquid face: every gas link is booked, and Σ ci falls by that sum.
    for (cx, cy, cz) in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        j = LatticeBoltzmann.src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
        flags[j] = TYPE_G
        ϕ[j] = 0f0
    end
    copyto!(domain.flags.data, flags)
    copyto!(domain.ϕ.data, ϕ)
    fill!(domain.ci.data, 0.2f0)
    fill!(domain.flux.data, 0f0)
    sum0 = sum(Array(domain.ci.data))
    _launch_surface0_even!(model)
    # Each of 6 links: old 0.2, outgoing 0.2, rec = 2*ceq − 0.2.
    δ_film = 6f0 * (0.2f0 - (2f0 * ceq - 0.2f0))
    @test Array(domain.flux.data)[1] ≈ δ_film atol=1f-4
    @test all(iszero, @view Array(domain.flux.data)[2:end])
    @test sum(Array(domain.ci.data)) ≈ sum0 - δ_film atol=1f-3

    domain.tag.data[n] = Int32(0)
    fill!(domain.ci.data, 0.2f0)
    fill!(domain.flux.data, 0f0)
    _launch_surface0_even!(model)
    @test all(iszero, Array(domain.flux.data))
end

# α = 0.01 is the metal diffusivity (ω_T ≈ 1.92). A frozen interface cell
# on the mold picked up an odd-even mode and sat near 0.80 while the wall
# was 0.90, about 60 K low on the aluminum scale. The wall is the floor.
@testset "frozen pore on a slow-cooling mold stays above the wall" begin
    Nx, Ny, Nz = 16, 16, 24
    H = 16
    Tmold = 0.899f0
    Tinit = 1.146f0
    model = Model(Nx, Ny, Nz, 0.2; backend=CPU(), σ=0.02, n_hydro=1,
                  α=0.01, Λ=0.406, Ts=0.955, Tl=1.0, K0=1.0f-4,
                  T_avg=Tinit, fz=-2.0f-5)
    domain = model.domains[1]
    flags = fill(TYPE_G, Nx * Ny * Nz)
    Th = fill(Tinit, length(flags))
    fsh = zeros(Float32, length(flags))
    for z in 0:(Nz - 1), y in 0:(Ny - 1), x in 0:(Nx - 1)
        n = 1 + x + Nx * y + Nx * Ny * z
        if x == 0 || y == 0 || z == 0 || x == Nx - 1 || y == Ny - 1 || z == Nz - 1
            flags[n] = TYPE_S | TYPE_T
            Th[n] = Tmold
            fsh[n] = 1
        elseif z < H
            flags[n] = TYPE_F
        end
    end
    copyto!(domain.flags.data, flags)
    copyto!(domain.T.data, Th)
    copyto!(domain.fs.data, fsh)
    set_foam!(model; D=0.01, k_H=1.0f-5, k_Π=0.02, q=0, V_m=30, γ_b=1, c0=0.05, ρ_liquid=1)
    nucleate_bubbles!(model, [(5.0, 5.0, 6.0), (11.0, 11.0, 6.0)], [2.5, 2.5])
    initialize!(model)
    for _ in 1:1500
        LatticeBoltzmann.step!(model)
    end
    tmin, _, _, twall, nmet = _metal_T_range(domain)
    @info "slow mold floor" tmin twall nmet
    @test nmet > 0
    @test twall ≈ Float64(Tmold) atol=1e-4
    # The example rejects 30 K, which is 0.034 on this scale. The odd-even
    # mode used to land 0.08 under the wall.
    @test tmin >= Float64(Tmold) - 0.02
    @test all(isfinite, Array(domain.T.data))
end

@testset "spawn after initialize adds a local dissolved source" begin
    Nx, Ny, Nz = 20, 20, 20
    model = Model(Nx, Ny, Nz, 0.2; backend=CPU(), σ=0.01, fz=0.0)
    domain = model.domains[1]
    N = Nx * Ny * Nz
    flags = fill(TYPE_G, N)
    for z in 0:(Nz - 1), y in 0:(Ny - 1), x in 0:(Nx - 1)
        n = 1 + x + Nx * y + Nx * Ny * z
        if x == 0 || y == 0 || z == 0 || x == Nx - 1 || y == Ny - 1 || z == Nz - 1
            flags[n] = TYPE_S | TYPE_T
        elseif z < 14
            flags[n] = TYPE_F
        end
    end
    copyto!(domain.flags.data, flags)
    copyto!(domain.T.data, fill(1.1f0, N))
    copyto!(domain.fs.data, zeros(Float32, N))
    set_foam!(model; D=0.01, k_H=1.0f-5, k_Π=0, q=0, V_m=3, γ_b=1, c0=0, ρ_liquid=1)
    initialize!(model)
    @test bubble_count(model) == 0
    @test all(iszero, Array(domain.c.data))

    δ = zeros(Float32, N)
    nsrc = 1 + 8 + Nx * 8 + Nx * Ny * 6
    δ[nsrc] = 0.05f0
    # A wall cell must not take solute.
    δ[1] = 0.2f0
    add_dissolved!(model, δ)
    c0 = Array(domain.c.data)
    @test c0[nsrc] ≈ 0.05f0
    @test c0[1] == 0
    LatticeBoltzmann.step!(model)
    @test sum(Array(domain.c.data)) ≈ 0.05 atol=1e-3

    ids = spawn_bubbles!(model, [(8.0, 8.0, 6.0)], [2.2])
    @test length(ids) == 1
    @test bubble_count(model) == 1
    flags1 = Array(domain.flags.data)
    tags = Array(domain.tag.data)
    nG = 0
    nblock = 0
    for z in 0:(Nz - 1), y in 0:(Ny - 1), x in 0:(Nx - 1)
        n = 1 + x + Nx * y + Nx * Ny * z
        su = flags1[n] & TYPE_SU
        if su == TYPE_G && tags[n] == ids[1]
            nG += 1
        end
        # Class-0 blockers are liquid again. The free surface sits near z = 14.
        if z < 10 && su == TYPE_I && tags[n] == 0
            nblock += 1
        end
    end
    @test nG >= 1
    @test nblock == 0
    LatticeBoltzmann.step!(model)
    @test bubble_ids(model) == ids
    @test bubble_volume(model, ids[1]) > 1
    @test all(isfinite, Array(domain.ρ.data))
    @test all(isfinite, Array(domain.T.data))
    @test all(isfinite, Array(domain.c.data))
end
