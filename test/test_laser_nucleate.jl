using Test
using LatticeBoltzmann
using KernelAbstractions
using Logging

@inline lbm_n_n4(x, y, z, Nx, Ny) = x + (y - 1) * Nx + (z - 1) * Nx * Ny

function nucleate_pool_model(; Nx=24, Ny=24, Nz=32, Hfill=20)
    @test SURFACE && TEMPERATURE
    si_H = 0.002
    Lpad = max(4, Hfill - 2)
    m = si_H / Lpad
    st = 0.1 * m^2 / 7.5e-6
    units = Units(Lpad, 0.05, 1, si_H, 0.05 * m / st, 8000.0; T=Float32,
                  K=1673.0f0, cp=500.0f0)
    nuc = Nucleation{Float32}(; enabled=false, d_min=6, R=1, c_star=1.10f0,
                              p_cell=1, n_max=3, n_over=1.0f0, every=1,
                              n_total_max=8)
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, α_c=0.2f0, k_H=3.0f0, fz=0, σ=0.02f0,
                  Λ=0.45f0, Ts=1.0f0, Tl=1.0f0, K0=1.0f-3, T_avg=0.97f0,
                  backend=CPU(), workgroup=64,
                  bubbles=BubbleTracker{Float32}(; nucleation=nuc, k_Π=0.08f0, d_max=4))
    model.units = units
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(0.97f0, Nx * Ny * Nz)
    fsh = ones(Float32, Nx * Ny * Nz)
    ch = fill(1.40f0, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_n4(x, y, z, Nx, Ny)
        if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
            host[n] = TYPE_S
            fsh[n] = 1
            ch[n] = 3.0f0 * P_ATM_LAT
        elseif z <= Hfill
            host[n] = TYPE_F
        else
            host[n] = TYPE_G
            fsh[n] = 0
            ch[n] = 3.0f0 * P_ATM_LAT
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].fs.data, fsh)
    copyto!(model.domains[1].c.data, ch)
    LatticeBoltzmann.initialize!(model)
    return model
end

function nucleate_pool_stats(model)
    d = model.domains[1]
    fl = Array(d.flags.data)
    TA = Array(d.T.data)
    uA = Array(d.u.data)
    fsA = Array(d.fs.data)
    Nx, Ny, Nz = Int(d.Nx), Int(d.Ny), Int(d.Nz)
    nF = nI = nliq = 0
    umax = 0.0f0
    Tmax = -Inf32
    @inbounds for n in eachindex(fl)
        su = fl[n] & TYPE_SU
        su == TYPE_F && (nF += 1)
        su == TYPE_I && (nI += 1)
        (su == TYPE_F || su == TYPE_I) || continue
        Tmax = max(Tmax, TA[n])
        fsA[n] < 0.5f0 && (nliq += 1)
        umax = max(umax, hypot(uA[n, 1], uA[n, 2], uA[n, 3]))
    end
    xc = (Nx + 1) ÷ 2
    yc = (Ny + 1) ÷ 2
    depth = liquid_column_depth(model, xc, yc)
    fm = foam_metrics(model)
    zs = Int[]
    B = model.bubbles
    if B isa BubbleTracker
        @inbounds for i in eachindex(B.label)
            B.label[i] > 0 || continue
            push!(zs, (i - 1) ÷ (Nx * Ny) + 1)
        end
    end
    zbar = isempty(zs) ? 0.0 : sum(zs) / length(zs)
    return (; nF, nI, nliq, umax, Tmax, depth, n_planted=fm.n_planted, nb=fm.nb, zbar)
end

@testset "nuclei form in a laser-melted pool without collapsing the pad" begin
    Nx, Ny, Nz = 24, 24, 32
    Hfill = 20
    model = nucleate_pool_model(; Nx, Ny, Nz, Hfill)
    xc = Float32(Nx + 1) / 2
    yc = Float32(Ny + 1) / 2
    model.laser = Laser(model.units; P=180.0, w=2.0, x=xc, y=yc,
                        z=Float32(Nz) - 1.1f0, nrays=7, max_bounce=4, skin=12)
    with_logger(NullLogger()) do
        run!(model, 240)
    end
    s1 = nucleate_pool_stats(model)
    @info "pre-nucleate pool" s1
    @test s1.depth >= 12
    @test s1.n_planted == 0
    model.bubbles.nucleation.enabled = true
    nF1 = s1.nF
    cores = nucleate_bubbles!(model, model.domains[1]; force=true)
    update_bubbles!(model)
    @test !isempty(cores)
    # Keep the beam on at cavity-test power after the plant.
    model.laser = Laser(model.units; P=20.0, w=2.0, x=xc, y=yc,
                        z=Float32(Nz) - 1.1f0, nrays=7, max_bounce=4, skin=12)
    with_logger(NullLogger()) do
        run!(model, 8)
    end
    s2 = nucleate_pool_stats(model)
    @info "post-nucleate pool" s2
    @test s2.n_planted >= 1
    @test s2.nb >= 1
    @test s2.zbar > 4
    @test s2.zbar < Hfill - 1
    @test s2.umax < 0.50f0
    @test s2.nF > 0.85 * nF1
    # 3³ G-cube splits the F column; depth is allowed to drop.
    @test s2.depth >= 3
    @test isfinite(s2.Tmax) && s2.Tmax < 8
end
