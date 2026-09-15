using Test
using LatticeBoltzmann
using KernelAbstractions
using Logging

@inline function lbm_n_b(x, y, z, Nx, Ny)
    return x + (y - 1) * Nx + (z - 1) * Nx * Ny
end

function paint_box_with_spheres!(host, Nx, Ny, Nz, spheres)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_b(x, y, z, Nx, Ny)
        if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
            host[n] = TYPE_S
            continue
        end
        g = false
        for (xc, yc, zc, R) in spheres
            dx, dy, dz = Float32(x) - xc, Float32(y) - yc, Float32(z) - zc
            g = g || (dx * dx + dy * dy + dz * dz <= R * R)
        end
        host[n] = g ? TYPE_G : TYPE_F
    end
    return host
end

function closed_bubble_model(Nx, Ny, Nz, spheres; σ=0.0f0, ν=0.1f0, bubbles=true)
    model = Model(Nx, Ny, Nz, ν; α=0.2f0, β=0.0f0, fz=0.0f0, σ=σ, Λ=0.0f0,
                  T_avg=1.0f0, backend=CPU(), workgroup=64,
                  bubbles = bubbles ? BubbleTracker{Float32}() : nothing)
    host = zeros(UInt8, Nx * Ny * Nz)
    paint_box_with_spheres!(host, Nx, Ny, Nz, spheres)
    copyto!(model.domains[1].flags.data, host)
    return model
end

@testset "Young–Laplace formula" begin
    σ, R, patm = 0.02f0, 5.0f0, P_ATM_LAT
    @test young_laplace_p(σ, R, patm) ≈ patm + 2 * σ / R
    @test bubble_radius(4 * π * R^3 / 3) ≈ R atol=1.0f-5
end

@testset "PLIC ρ_gas follows p_id" begin
    ϕ = fill(0.5f0, 5 * 5 * 5)
    @test LatticeBoltzmann.gas_density_plic(0.0f0, ϕ, 0.5f0, 2, 2, 2, 5, 5, 5) ≈ 1.0f0
    @test LatticeBoltzmann.gas_density_plic(0.0f0, ϕ, 0.5f0, 2, 2, 2, 5, 5, 5, 2.0f0 / 3.0f0) ≈ 2.0f0
end

@testset "open headroom is atmosphere, not a bubble" begin
    @test SURFACE
    Nx, Ny, Nz = 12, 12, 16
    Hfill = 8
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, backend=CPU(), workgroup=64,
                  bubbles=BubbleTracker{Float32}())
    host = zeros(UInt8, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_b(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
        else
            host[n] = TYPE_G
        end
    end
    copyto!(model.domains[1].flags.data, host)
    LatticeBoltzmann.initialize!(model)
    @test model.bubbles.nb == 0
    pg = Array(model.domains[1].p_gas.data)
    @test all(x -> abs(x - P_ATM_LAT) < 1.0f-6, pg)
end

@testset "enclosed cavity is one bubble at p_atm" begin
    @test SURFACE
    Nx = Ny = Nz = 20
    xc = yc = zc = (Nx + 1) / 2
    R = 4.0f0
    model = closed_bubble_model(Nx, Ny, Nz, ((xc, yc, zc, R),))
    LatticeBoltzmann.initialize!(model)
    recs = bubble_records(model)
    @test length(recs) == 1
    b = recs[1]
    @test b.V > 10
    @test b.R > 2
    @test abs(b.p - P_ATM_LAT) / P_ATM_LAT < 0.05f0
    @test b.n ≈ b.p * b.V atol=0.05f0
end

@testset "coalescence conserves n" begin
    @test SURFACE
    Nx, Ny, Nz = 28, 16, 16
    zc = yc = (Ny + 1) / 2
    R = 3.5f0
    model = closed_bubble_model(Nx, Ny, Nz, ((8.0f0, yc, zc, R), (20.0f0, yc, zc, R)))
    LatticeBoltzmann.initialize!(model)
    recs = bubble_records(model)
    @test length(recs) == 2
    set_bubble_n!(model, recs[1].id, 1.0f0)
    set_bubble_n!(model, recs[2].id, 2.0f0)
    n0 = 3.0f0
    flags = Array(model.domains[1].flags.data)
    yb, zb = round(Int, yc), round(Int, zc)
    for x in 8:20
        n = lbm_n_b(x, yb, zb, Nx, Ny)
        (flags[n] & TYPE_S) == 0x00 && (flags[n] = TYPE_G)
    end
    copyto!(model.domains[1].flags.data, flags)
    update_bubbles!(model)
    recs2 = bubble_records(model)
    @test length(recs2) == 1
    @test recs2[1].n ≈ n0 rtol=0.08
end

@testset "overpressure grows, underpressure shrinks" begin
    @test SURFACE
    Nx = Ny = 20
    Nz = 32
    xc = yc = (Nx + 1) / 2
    zc = 12.0f0
    R = 4.0f0
    spheres = ((xc, yc, zc, R),)

    model_hi = closed_bubble_model(Nx, Ny, Nz, spheres; σ=0.0f0)
    LatticeBoltzmann.initialize!(model_hi)
    rec0 = bubble_records(model_hi)
    @test length(rec0) == 1
    b0 = rec0[1]
    set_bubble_n!(model_hi, b0.id, 1.8f0 * P_ATM_LAT * b0.V)
    V0h = b0.V
    with_logger(NullLogger()) do
        run!(model_hi, 8)
    end
    rech = bubble_records(model_hi)
    @test length(rech) == 1
    Vh = rech[1].V

    model_lo = closed_bubble_model(Nx, Ny, Nz, spheres; σ=0.0f0)
    LatticeBoltzmann.initialize!(model_lo)
    rec1 = bubble_records(model_lo)
    @test length(rec1) == 1
    b1 = rec1[1]
    set_bubble_n!(model_lo, b1.id, 0.45f0 * P_ATM_LAT * b1.V)
    V0l = b1.V
    with_logger(NullLogger()) do
        run!(model_lo, 8)
    end
    recl = bubble_records(model_lo)
    @test !isempty(recl)
    Vl = sum(r -> r.V, recl)

    @info "pV bubble hydro" V0h Vh V0l Vl
    @test Vh > V0h * 1.05f0
    @test Vl < V0l * 0.95f0
    @test Vh > Vl
end

@testset "Laplace-loaded bubble holds its radius" begin
    @test SURFACE
    Nx = Ny = 20
    Nz = 32
    xc = yc = (Nx + 1) / 2
    zc = 12.0f0
    R0 = 4.0f0
    σ = 0.02f0
    model = closed_bubble_model(Nx, Ny, Nz, ((xc, yc, zc, R0),); σ=σ)
    LatticeBoltzmann.initialize!(model)
    b0 = bubble_records(model)[1]
    pL0 = young_laplace_p(σ, b0.R, P_ATM_LAT)
    set_bubble_n!(model, b0.id, pL0 * b0.V)
    bset = bubble_records(model)[1]
    @test bset.p ≈ pL0 rtol=0.02
    with_logger(NullLogger()) do
        run!(model, 20)
    end
    recs = bubble_records(model)
    @test length(recs) == 1
    b = recs[1]
    @info "Laplace bubble" R0=b0.R R=b.R p=b.p pL0 n=b.n V=b.V
    @test b.R ≈ b0.R rtol=0.2
end
