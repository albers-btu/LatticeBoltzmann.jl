using Test
using LatticeBoltzmann
using KernelAbstractions
using Logging

@inline function lbm_n_d(x, y, z, Nx, Ny)
    return x + (y - 1) * Nx + (z - 1) * Nx * Ny
end

function paint_liquid_box!(host, Nx, Ny, Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_d(x, y, z, Nx, Ny)
        host[n] = (x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz) ?
            TYPE_S : TYPE_F
    end
    return host
end

@testset "dissolved Gaussian spreads as 2Dt" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 48, 8, 8
    α_c = 0.2f0
    D = α_c / 2
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, α_c=α_c, k_H=0, fz=0, σ=0,
                  backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    paint_liquid_box!(host, Nx, Ny, Nz)
    ch = ones(Float32, Nx * Ny * Nz)
    xc = (Nx + 1) / 2
    s0 = 3.0f0
    A = 0.4f0
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_d(x, y, z, Nx, Ny)
        (host[n] & TYPE_S) != 0 && continue
        dx = Float32(x) - xc
        ch[n] = 1 + A * exp(-dx * dx / (2 * s0 * s0))
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].c.data, ch)
    LatticeBoltzmann.initialize!(model)
    nsteps = 80
    with_logger(NullLogger()) do
        run!(model, nsteps)
    end
    cA = Array(model.domains[1].c.data)
    fl = Array(model.domains[1].flags.data)
    m1 = 0.0; m2 = 0.0; w = 0.0
    for z in 2:(Nz - 1), y in 2:(Ny - 1), x in 2:(Nx - 1)
        n = lbm_n_d(x, y, z, Nx, Ny)
        (fl[n] & TYPE_S) != 0 && continue
        dc = max(Float64(cA[n]) - 1, 0)
        m1 += dc * x
        m2 += dc * x * x
        w += dc
    end
    var = m2 / w - (m1 / w)^2
    var_an = s0 * s0 + 2 * D * nsteps
    @info "dissolved Gaussian" var var_an D nsteps
    @test w > 0
    @test var > s0 * s0
    # D3Q7 AA (same as T) spreads faster than continuum 2Dt; keep it bounded.
    @test var > 0.5 * var_an
    @test var < 2.5 * var_an
end

@testset "Henry sets c_I = k_H p on a bubble" begin
    @test SURFACE
    Nx = Ny = Nz = 20
    xc = yc = zc = (Nx + 1) / 2
    R = 4.0f0
    k_H = 3.0f0
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, α_c=0.2f0, k_H=k_H, fz=0, σ=0,
                  backend=CPU(), workgroup=64, bubbles=BubbleTracker{Float32}())
    host = zeros(UInt8, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_d(x, y, z, Nx, Ny)
        if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
            host[n] = TYPE_S
        else
            dx, dy, dz = Float32(x) - xc, Float32(y) - yc, Float32(z) - zc
            host[n] = dx * dx + dy * dy + dz * dz <= R * R ? TYPE_G : TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    fill!(model.domains[1].c.data, 1.25f0)
    LatticeBoltzmann.initialize!(model)
    with_logger(NullLogger()) do
        run!(model, 8)
    end
    cA = Array(model.domains[1].c.data)
    fl = Array(model.domains[1].flags.data)
    pg = Array(model.domains[1].p_gas.data)
    bd = Array(model.domains[1].bid.data)
    err = 0.0; nI = 0
    for n in eachindex(cA)
        (fl[n] & TYPE_SU) == TYPE_I || continue
        bd[n] < 0.5f0 && continue
        nI += 1
        err += abs(Float64(cA[n]) - Float64(k_H * pg[n]))
    end
    @info "Henry interface" nI mean_abs=err / max(nI, 1)
    @test nI > 10
    @test err / nI < 0.15
end

@testset "supersaturation grows n; undersaturation shrinks n" begin
    @test SURFACE
    function run_sat(c∞)
        Nx = Ny = 18
        Nz = 24
        xc = yc = (Nx + 1) / 2
        zc = 10.0f0
        R = 4.5f0
        k_H = 3.0f0
        model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, α_c=0.2f0, k_H=k_H, fz=0, σ=0,
                      backend=CPU(), workgroup=64, bubbles=BubbleTracker{Float32}())
        host = zeros(UInt8, Nx * Ny * Nz)
        for z in 1:Nz, y in 1:Ny, x in 1:Nx
            n = lbm_n_d(x, y, z, Nx, Ny)
            if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
                host[n] = TYPE_S
            else
                dx, dy, dz = Float32(x) - xc, Float32(y) - yc, Float32(z) - zc
                host[n] = dx * dx + dy * dy + dz * dz <= R * R ? TYPE_G : TYPE_F
            end
        end
        copyto!(model.domains[1].flags.data, host)
        fill!(model.domains[1].c.data, Float32(c∞))
        LatticeBoltzmann.initialize!(model)
        n0 = bubble_records(model)[1].n
        with_logger(NullLogger()) do
            run!(model, 8)
        end
        recs = bubble_records(model)
        return n0, isempty(recs) ? 0.0f0 : recs[1].n
    end
    n0h, nh = run_sat(1.4f0)
    n0l, nl = run_sat(0.6f0)
    @info "Henry n flux" n0h nh n0l nl
    @test nh > n0h * 1.02f0
    @test nl < n0l * 0.98f0
end

@testset "Epstein–Plesset R² vs 2 D (c∞/k_H − p)/p t" begin
    @test SURFACE
    Nx = Ny = Nz = 28
    xc = yc = zc = (Nx + 1) / 2
    R0_paint = 5.0f0
    k_H = 3.0f0
    α_c = 0.2f0
    D = α_c / 2
    c∞ = 1.35f0
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, α_c=α_c, k_H=k_H, fz=0, σ=0.0f0,
                  backend=CPU(), workgroup=64, bubbles=BubbleTracker{Float32}())
    host = zeros(UInt8, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_d(x, y, z, Nx, Ny)
        if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
            host[n] = TYPE_S
        else
            dx, dy, dz = Float32(x) - xc, Float32(y) - yc, Float32(z) - zc
            host[n] = dx * dx + dy * dy + dz * dz <= R0_paint * R0_paint ? TYPE_G : TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    fill!(model.domains[1].c.data, c∞)
    LatticeBoltzmann.initialize!(model)
    b0 = bubble_records(model)[1]
    nsteps = 60
    with_logger(NullLogger()) do
        run!(model, nsteps)
    end
    recs = bubble_records(model)
    @test length(recs) == 1
    b = recs[1]
    R2an = epstein_plesset_R2(b0.R, nsteps, D, c∞, k_H, P_ATM_LAT)
    @info "Epstein–Plesset" R0=b0.R R=b.R R2=b.R^2 R2an n=b.n n0=b0.n
    @test b.R > b0.R
    @test isapprox(b.R^2, R2an; rtol=0.45)
end
