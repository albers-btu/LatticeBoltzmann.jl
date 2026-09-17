using Test
using LatticeBoltzmann
using KernelAbstractions
using Logging

@inline lbm_n_p(x, y, z, Nx, Ny) = x + (y - 1) * Nx + (z - 1) * Nx * Ny

function pool_pad_model(; Nx=24, Ny=24, Nz=32, Hfill=20, σ=0.02f0,
                        Tpad=0.97f0, solid=true)
    @test SURFACE && TEMPERATURE
    si_H = 0.002
    Lpad = max(4, Hfill - 2)
    m = si_H / Lpad
    st = 0.1 * m^2 / 7.5e-6
    units = Units(Lpad, 0.05, 1, si_H, 0.05 * m / st, 8000.0; T=Float32,
                  K=1673.0f0, cp=500.0f0)
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=σ, Λ=0.45f0,
                  Ts=1.0f0, Tl=1.0f0, K0=1.0f-3, T_avg=Tpad,
                  backend=CPU(), workgroup=64)
    model.units = units
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Tpad, Nx * Ny * Nz)
    fsh = fill(solid ? 1.0f0 : 0.0f0, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_p(x, y, z, Nx, Ny)
        if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
            host[n] = TYPE_S
            fsh[n] = 1
        elseif z <= Hfill
            host[n] = TYPE_F
        else
            host[n] = TYPE_G
            fsh[n] = 0
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].fs.data, fsh)
    LatticeBoltzmann.initialize!(model)
    return model
end

function pool_stats(model)
    d = model.domains[1]
    fl = Array(d.flags.data)
    TA = Array(d.T.data)
    uA = Array(d.u.data)
    fsA = Array(d.fs.data)
    nF = nI = nliq = 0
    umax = 0.0f0
    Tmax = -Inf32
    finite = true
    @inbounds for n in eachindex(fl)
        su = fl[n] & TYPE_SU
        su == TYPE_F && (nF += 1)
        su == TYPE_I && (nI += 1)
        (su == TYPE_F || su == TYPE_I) || continue
        finite &= isfinite(TA[n])
        Tmax = max(Tmax, TA[n])
        fsA[n] < 0.5f0 && (nliq += 1)
        umax = max(umax, hypot(uA[n, 1], uA[n, 2], uA[n, 3]))
    end
    xc = (Int(d.Nx) + 1) ÷ 2
    yc = (Int(d.Ny) + 1) ÷ 2
    depth = liquid_column_depth(model, xc, yc)
    return (; nF, nI, nliq, umax, Tmax, finite, depth)
end

@testset "liquid_column_depth counts a painted liquid stack" begin
    model = pool_pad_model(; solid=false, Tpad=1.05f0)
    s0 = pool_stats(model)
    @info "painted liquid depth" s0
    @test s0.depth >= 12
end

@testset "laser melts a solid pad to ≥12 liquid cells on the axis" begin
    Nx, Ny, Nz = 24, 24, 32
    Hfill = 20
    model = pool_pad_model(; Nx, Ny, Nz, Hfill, solid=true, Tpad=0.97f0, σ=0.02f0)
    s0 = pool_stats(model)
    @test s0.depth == 0
    @test s0.nliq < 50
    @test model.bubbles === nothing
    xc = Float32(Nx + 1) / 2
    yc = Float32(Ny + 1) / 2
    model.laser = Laser(model.units; P=180.0, w=2.0, x=xc, y=yc,
                        z=Float32(Nz) - 1.1f0, nrays=7, max_bounce=4, skin=12)
    with_logger(NullLogger()) do
        run!(model, 240)
    end
    s1 = pool_stats(model)
    @info "melted pool" s0 s1
    @test model.bubbles === nothing
    @test s1.finite
    @test isfinite(s1.Tmax) && s1.Tmax > 1.0f0 && s1.Tmax < 20
    @test s1.depth >= 12
    @test s1.nliq > s0.nliq + 100
    @test s1.umax < 0.35f0
    @test s1.nF > 0.85 * s0.nF
end
