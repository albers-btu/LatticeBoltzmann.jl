using Test
using LatticeBoltzmann
using KernelAbstractions
using CUDA
using Random
using Logging

# Closed liquid cube, no atmosphere. 8 → 20 → 40 Poisson-disk R=5 seeds,
# Fig. 15 knobs. Voxel CSF still Mach-outs by t=200; 8-seed grows through t=100.

@inline lbm_n_cf(x, y, z, Nx, Ny) = x + (y - 1) * Nx + (z - 1) * Nx * Ny

function closed_foam_brief(model)
    recs = bubble_records(model)
    V = 0.0
    Rmax = 0.0
    @inbounds for r in recs
        V += Float64(r.V)
        Rmax = max(Rmax, Float64(r.R))
    end
    d = model.domains[1]
    uA = Array(d.u.data)
    fl = Array(d.flags.data)
    umax = 0.0f0
    nF = 0
    nI = 0
    nG = 0
    @inbounds for n in axes(uA, 1)
        su = fl[n] & TYPE_SU
        if su == TYPE_F
            nF += 1
            umax = max(umax, hypot(uA[n, 1], uA[n, 2], uA[n, 3]))
        elseif su == TYPE_I
            nI += 1
            umax = max(umax, hypot(uA[n, 1], uA[n, 2], uA[n, 3]))
        elseif su == TYPE_G
            nG += 1
        end
    end
    return (; nb=length(recs), V, Rmax, umax, nF, nI, nG)
end

function closed_foam_model(; Nx, Ny, Nz, n_nuclei, R0=5.0f0, d_min=16.0f0, seed=15)
    ν = 0.15f0
    σ = 0.005f0
    k_Π = 0.005f0
    d_max = 4.0f0
    k_H = 3.0f0
    α_c = 0.015f0
    c0 = 1.20f0
    T_gas = 1.0f0
    backend = CUDA.functional() ? CUDABackend() : CPU()
    Random.seed!(seed)
    pad = 2 + ceil(Int, R0) + 3
    centers = poisson_disk_3d(
        n_nuclei, d_min,
        (1 + pad, Nx - pad),
        (1 + pad, Ny - pad),
        (1 + pad, Nz - pad),
    )
    model = Model(Nx, Ny, Nz, ν;
                  α=0.2f0, α_c=α_c, k_H=k_H, fz=0, σ=σ, T_avg=1.0f0,
                  backend=backend, workgroup=backend isa CUDABackend ? 256 : 64,
                  bubbles=BubbleTracker{Float32}(;
                      T_gas=T_gas, k_Π=k_Π, d_max=d_max, every=10,
                      nucleation=nothing))
    host = zeros(UInt8, Nx * Ny * Nz)
    paint_spherical_nuclei!(host, Nx, Ny, Nz, centers, R0)
    ch = fill(k_H * P_ATM_LAT, Nx * Ny * Nz)
    @inbounds for n in eachindex(host)
        (host[n] & TYPE_SU) == TYPE_F && (ch[n] = c0)
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].c.data, ch)
    LatticeBoltzmann.initialize!(model)
    equilibrate_nuclei_n!(model; n_over=1)
    return model, centers, backend
end

function run_closed_foam!(model, nsteps, tag)
    s0 = closed_foam_brief(model)
    @info "closed-box $tag t=0" t=Int(model.domains[1].t) s0.nb s0.V s0.Rmax s0.umax s0.nF s0.nI s0.nG
    snapshots = [s0]
    for _ in 50:50:nsteps
        with_logger(NullLogger()) do
            run!(model, 50)
        end
        s = closed_foam_brief(model)
        push!(snapshots, s)
        @info "closed-box $tag" t=Int(model.domains[1].t) s.nb s.V s.Rmax s.umax s.nF s.nI s.nG
    end
    return snapshots
end

const CLOSED_FOAM_CASES = (
    (tag="8-seed",  n_nuclei=8,  Nx=64, nsteps=200),
    (tag="20-seed", n_nuclei=20, Nx=72, nsteps=200),
    (tag="40-seed", n_nuclei=40, Nx=88, nsteps=200),
)

@testset "closed-box quieter seeds" begin
    @testset "$(c.tag)" for c in CLOSED_FOAM_CASES
        @test SURFACE && TEMPERATURE
        Nx = c.Nx
        model, centers, backend = closed_foam_model(;
            Nx=Nx, Ny=Nx, Nz=Nx, n_nuclei=c.n_nuclei)
        n_nuclei = length(centers)
        @test n_nuclei == c.n_nuclei
        snaps = run_closed_foam!(model, c.nsteps, c.tag)
        s0 = snaps[1]
        s = snaps[end]
        @info "closed-box $(c.tag) done" backend s0.nb s0.V s0.umax s.nb s.V s.Rmax s.umax
        @test s0.nb == n_nuclei
        @test s0.V > 0
        @test s0.umax < 0.2f0
        @test s.nb >= max(1, n_nuclei - 4)
        @test s.nb <= n_nuclei + 5
        @test s.nF > 0.5 * s0.nF
        s50 = snaps[2]
        if c.n_nuclei <= 20
            @test s50.V > s0.V
        end
        if c.n_nuclei == 8 && length(snaps) >= 3
            s100 = snaps[3]
            @test s100.V > s0.V
            @test s100.umax < 0.3f0
        end
        # Voxel CSF: 8-seed is quiet through t=100, Mach-out by t=200.
        @test_broken s.umax < 0.3f0
    end
end
