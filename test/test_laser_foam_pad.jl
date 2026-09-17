using Test
using LatticeBoltzmann
using KernelAbstractions
using Logging

@inline lbm_n_c6(x, y, z, Nx, Ny) = x + (y - 1) * Nx + (z - 1) * Nx * Ny

@testset "powder + Arrhenius + nuclei under full-power laser" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 24, 24, 32
    Hfill = 20
    si_H = 0.002
    Lpad = Hfill - 2
    m = si_H / Lpad
    st = 0.1 * m^2 / 7.5e-6
    units = Units(Lpad, 0.05, 1, si_H, 0.05 * m / st, 8000.0; T=Float32,
                  K=1673.0f0, cp=500.0f0)
    nuc = Nucleation{Float32}(; enabled=false, d_min=6, R=1, c_star=1.05f0,
                              p_cell=1, n_max=2, n_over=1.0f0, every=1,
                              n_total_max=6)
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, α_c=0, k_H=3.0f0, fz=0, σ=0.02f0,
                  k_a=0.08f0, E_a=4.0f0, Y_a=1, a_fs_max=0.5f0,
                  Λ=0.45f0, Ts=1.0f0, Tl=1.0f0, K0=1.0f-3, T_avg=0.97f0,
                  τ_p=0, T_p=0.97f0, backend=CPU(), workgroup=64,
                  bubbles=BubbleTracker{Float32}(; nucleation=nuc, k_Π=0.08f0, d_max=4))
    model.units = units
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(0.97f0, Nx * Ny * Nz)
    fsh = ones(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_c6(x, y, z, Nx, Ny)
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
    xc = Float32(Nx + 1) / 2
    yc = Float32(Ny + 1) / 2
    P = 180.0
    model.laser = Laser(units; P=P, w=2.0, x=xc, y=yc,
                        z=Float32(Nz) - 1.1f0, nrays=7, max_bounce=4, skin=12)
    model.powder_jet = PowderJet(units; mdot=5.0e-5, w=2.0, v=3.0,
                                 x=xc, y=yc, z=Float32(Nz) - 1.2f0,
                                 dir=(0, 0, -1), nparcels=8, agent_frac=0.4)
    LatticeBoltzmann.initialize!(model)
    with_logger(NullLogger()) do
        run!(model, 240)
    end
    depth1 = liquid_column_depth(model, (Nx + 1) ÷ 2, (Ny + 1) ÷ 2)
    fm1 = foam_metrics(model)
    cA = Array(model.domains[1].c.data)
    fsA = Array(model.domains[1].fs.data)
    fl = Array(model.domains[1].flags.data)
    c_liq = 0.0f0
    @inbounds for n in eachindex(fl)
        su = fl[n] & TYPE_SU
        (su == TYPE_F || su == TYPE_I) && fsA[n] < 0.5f0 && (c_liq = max(c_liq, cA[n]))
    end
    nF1 = count(n -> (fl[n] & TYPE_SU) == TYPE_F, eachindex(fl))
    @info "combined pre-nuc" depth1 c_liq a=fm1.a res=fm1.a_res planted=fm1.n_planted
    @test depth1 >= 8
    @test c_liq > 1.05f0
    @test fm1.n_planted == 0
    @test fm1.a + fm1.a_res > 0.05
    # Agent dumped at the free surface; mix that supersaturation into the melt
    # so a 3³ F cube can nucleate (no dissolved D3Q7 in this test).
    cA = Array(model.domains[1].c.data)
    @inbounds for n in eachindex(fl)
        su = fl[n] & TYPE_SU
        (su == TYPE_F || su == TYPE_I) || continue
        fsA[n] < 0.5f0 || continue
        cA[n] = max(cA[n], 1.25f0)
    end
    copyto!(model.domains[1].c.data, cA)
    # Full power; shrink numerical skin so Q stays in the lid, not the 3³ hole.
    model.laser.skin = 4
    @test model.laser.P == P
    model.bubbles.nucleation.enabled = true
    cores = nucleate_bubbles!(model, model.domains[1]; force=true)
    update_bubbles!(model)
    @test !isempty(cores)
    with_logger(NullLogger()) do
        run!(model, 8)
    end
    fl2 = Array(model.domains[1].flags.data)
    uA = Array(model.domains[1].u.data)
    TA = Array(model.domains[1].T.data)
    nF2 = 0
    umax = 0.0f0
    Tmax = -Inf32
    @inbounds for n in eachindex(fl2)
        su = fl2[n] & TYPE_SU
        su == TYPE_F && (nF2 += 1)
        (su == TYPE_F || su == TYPE_I) || continue
        umax = max(umax, hypot(uA[n, 1], uA[n, 2], uA[n, 3]))
        Tmax = max(Tmax, TA[n])
    end
    fm2 = foam_metrics(model)
    zs = Int[]
    @inbounds for i in eachindex(model.bubbles.label)
        model.bubbles.label[i] > 0 || continue
        push!(zs, (i - 1) ÷ (Nx * Ny) + 1)
    end
    zbar = isempty(zs) ? 0.0 : sum(zs) / length(zs)
    @info "combined post-nuc" fm2.n_planted fm2.nb zbar umax Tmax nF1 nF2 P=model.laser.P skin=model.laser.skin
    @test model.laser.P == P
    @test fm2.n_planted >= 1
    @test fm2.nb >= 1
    @test zbar > 4 && zbar <= Hfill + 1
    @test umax < 0.50f0
    @test nF2 > 0.85 * nF1
    @test isfinite(Tmax) && Tmax < 8
end
