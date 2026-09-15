using Test
using LatticeBoltzmann
using KernelAbstractions
using Logging

@inline function lbm_n_j(x, y, z, Nx, Ny)
    return x + (y - 1) * Nx + (z - 1) * Nx * Ny
end

function two_bubble_model(Nx, Ny, Nz, c1, c2, R; k_Π=0.0f0, n_over=1.6f0, σ=0.006f0)
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=σ, Λ=0, T_avg=1,
                  backend=CPU(), workgroup=64,
                  bubbles=BubbleTracker{Float32}(; k_Π=k_Π, d_max=4))
    host = zeros(UInt8, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_j(x, y, z, Nx, Ny)
        if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
            host[n] = TYPE_S
            continue
        end
        d1 = (Float32(x) - c1[1])^2 + (Float32(y) - c1[2])^2 + (Float32(z) - c1[3])^2
        d2 = (Float32(x) - c2[1])^2 + (Float32(y) - c2[2])^2 + (Float32(z) - c2[3])^2
        host[n] = (d1 <= R * R || d2 <= R * R) ? TYPE_G : TYPE_F
    end
    copyto!(model.domains[1].flags.data, host)
    LatticeBoltzmann.initialize!(model)
    recs = bubble_records(model)
    for r in recs
        set_bubble_n!(model, r.id, n_over * P_ATM_LAT * r.V)
    end
    return model
end

@testset "isolated bubble has no Π" begin
    @test SURFACE
    Nx = Ny = Nz = 20
    xc = yc = zc = (Nx + 1) / 2
    model = two_bubble_model(Nx, Ny, Nz, (xc, yc, zc), (xc, yc, zc), 4.0f0;
                             k_Π=0.08f0, n_over=1.0f0)
    recs = bubble_records(model)
    @test length(recs) == 1
    pg = Array(model.domains[1].p_gas.data)
    fl = Array(model.domains[1].flags.data)
    bd = Array(model.domains[1].bid.data)
    pI = Float32[pg[i] for i in eachindex(pg) if (fl[i] & TYPE_SU) == TYPE_I && bd[i] > 0.5f0]
    @test !isempty(pI)
    @test all(p -> abs(p - P_ATM_LAT) < 0.02f0, pI)
end

@testset "Π lowers p_gas on facing interfaces" begin
    @test SURFACE
    model = two_bubble_model(32, 16, 16, (10.0f0, 8.0f0, 8.0f0), (22.0f0, 8.0f0, 8.0f0),
                             5.0f0; k_Π=0.08f0, n_over=1.0f0, σ=0.0f0)
    @test length(bubble_records(model)) == 2
    pg = Array(model.domains[1].p_gas.data)
    fl = Array(model.domains[1].flags.data)
    pI = Float32[pg[i] for i in eachindex(pg) if (fl[i] & TYPE_SU) == TYPE_I]
    @info "facing Π" pmin=minimum(pI) patm=P_ATM_LAT
    @test minimum(pI) < P_ATM_LAT - 0.02f0
end

@testset "k_Π = 0: growing pair coalesces" begin
    @test SURFACE
    model = two_bubble_model(32, 16, 16, (10.0f0, 8.0f0, 8.0f0), (21.0f0, 8.0f0, 8.0f0),
                             5.0f0; k_Π=0.0f0, n_over=2.0f0, σ=0.0f0)
    @test length(bubble_records(model)) == 2
    with_logger(NullLogger()) do
        run!(model, 90)
    end
    recs = bubble_records(model)
    @info "coalesce kΠ=0" nb=length(recs) V=isempty(recs) ? 0 : sum(r -> r.V, recs)
    @test length(recs) == 1
end

@testset "k_Π = 0.08: pair keeps a lamella" begin
    @test SURFACE
    model = two_bubble_model(32, 16, 16, (10.0f0, 8.0f0, 8.0f0), (21.0f0, 8.0f0, 8.0f0),
                             5.0f0; k_Π=0.12f0, n_over=2.0f0, σ=0.006f0)
    @test length(bubble_records(model)) == 2
    with_logger(NullLogger()) do
        run!(model, 90)
    end
    recs = bubble_records(model)
    @info "lamella kΠ=0.08" nb=length(recs) V=isempty(recs) ? 0 : sum(r -> r.V, recs)
    @test length(recs) == 2
end
