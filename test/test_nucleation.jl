using Test
using LatticeBoltzmann
using KernelAbstractions
using Logging

@inline function lbm_n_n(x, y, z, Nx, Ny)
    return x + (y - 1) * Nx + (z - 1) * Nx * Ny
end

function liquid_box!(host, Nx, Ny, Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_n(x, y, z, Nx, Ny)
        host[n] = (x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz) ?
            TYPE_S : TYPE_F
    end
    return host
end

function nucleus_model(Nx, Ny, Nz; c∞=1.3f0, c_star=1.05f0, d_min=8, n_max=12,
                       k_H=3.0f0, α_c=0.2f0, n_over=1.2f0, every=1)
    nuc = Nucleation{Float32}(; d_min=d_min, R=1, c_star=c_star, p_cell=1,
                              n_max=n_max, n_over=n_over, every=every, n_total_max=64)
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, α_c=α_c, k_H=k_H, fz=0, σ=0,
                  backend=CPU(), workgroup=64,
                  bubbles=BubbleTracker{Float32}(; nucleation=nuc))
    host = zeros(UInt8, Nx * Ny * Nz)
    liquid_box!(host, Nx, Ny, Nz)
    copyto!(model.domains[1].flags.data, host)
    fill!(model.domains[1].c.data, Float32(c∞))
    return model
end

function min_centre_dist(flags, Nx, Ny, Nz)
    cores = NTuple{3,Int}[]
    for z in 2:(Nz - 1), y in 2:(Ny - 1), x in 2:(Nx - 1)
        n = lbm_n_n(x, y, z, Nx, Ny)
        (flags[n] & TYPE_SU) == TYPE_G || continue
        push!(cores, (x, y, z))
    end
    dmin = Inf
    for i in 1:length(cores), j in (i + 1):length(cores)
        a, b = cores[i], cores[j]
        d = sqrt(Float64((a[1] - b[1])^2 + (a[2] - b[2])^2 + (a[3] - b[3])^2))
        dmin = min(dmin, d)
    end
    return length(cores), dmin
end

@testset "no nucleation below c_star" begin
    @test SURFACE
    model = nucleus_model(24, 24, 24; c∞=0.9f0, c_star=1.05f0, n_max=20)
    LatticeBoltzmann.initialize!(model)
    cores = nucleate_bubbles!(model, model.domains[1]; force=true)
    @test isempty(cores)
    @test model.bubbles.nb == 0
    @test model.bubbles.nucleation.n_planted == 0
end

@testset "Poisson-disk nuclei respect d_min" begin
    @test SURFACE
    d_min = 7
    model = nucleus_model(28, 28, 28; c∞=1.4f0, c_star=1.05f0, d_min=d_min, n_max=16)
    LatticeBoltzmann.initialize!(model)
    cores = nucleate_bubbles!(model, model.domains[1]; force=true)
    update_bubbles!(model)
    @test length(cores) >= 3
    @test model.bubbles.nb == length(cores)
    fl = Array(model.domains[1].flags.data)
    nG, d = min_centre_dist(fl, 28, 28, 28)
    @info "Poisson disk" n_planted=length(cores) nG d d_min
    @test nG == length(cores)
    @test d + 1e-6 >= d_min
end

@testset "nuclei are enclosed bubbles with overpressure" begin
    @test SURFACE
    model = nucleus_model(20, 20, 20; c∞=1.35f0, c_star=1.05f0, d_min=8, n_max=4,
                          n_over=1.5f0)
    LatticeBoltzmann.initialize!(model)
    cores = nucleate_bubbles!(model, model.domains[1]; force=true)
    update_bubbles!(model)
    flags = Array(model.domains[1].flags.data)
    B = model.bubbles
    LatticeBoltzmann._seed_new_nuclei!(
        B, cores, flags, model.domains[1].p_gas.data, model.domains[1].bid.data,
        model.domains[1].σ, 20, 20, 20)
    recs = bubble_records(model)
    @test !isempty(recs)
    @test all(r -> r.V >= 1, recs)
    @test all(r -> r.p > P_ATM_LAT * 1.2f0, recs)
end

@testset "run! nucleates in supersaturated liquid" begin
    @test SURFACE
    model = nucleus_model(24, 24, 24; c∞=1.4f0, c_star=1.1f0, d_min=8, n_max=6,
                          every=5)
    LatticeBoltzmann.initialize!(model)
    @test model.bubbles.nb == 0
    with_logger(NullLogger()) do
        run!(model, 5)
    end
    @test model.bubbles.nucleation.n_planted >= 1
    @test model.bubbles.nb >= 1
end
