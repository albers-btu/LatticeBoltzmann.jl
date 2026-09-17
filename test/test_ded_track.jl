using Test
using LatticeBoltzmann
using KernelAbstractions
using Logging

@inline lbm_n_t7(x, y, z, Nx, Ny) = x + (y - 1) * Nx + (z - 1) * Nx * Ny

function mix_liquid_c_track!(model, cmin=1.25f0)
    d = model.domains[1]
    cA = Array(d.c.data)
    fsA = Array(d.fs.data)
    fl = Array(d.flags.data)
    @inbounds for n in eachindex(cA)
        su = fl[n] & TYPE_SU
        (su == TYPE_F || su == TYPE_I) || continue
        # Nucleation uses is_solid_fraction (fs ≳ 0.999); mix the mushy pool too.
        fsA[n] < 0.99f0 || continue
        cA[n] = max(cA[n], cmin)
    end
    copyto!(d.c.data, cA)
    return nothing
end

function trail_span(model, y)
    d = model.domains[1]
    Nx, Ny, Nz = Int(d.Nx), Int(d.Ny), Int(d.Nz)
    fl = Array(d.flags.data)
    fsA = Array(d.fs.data)
    yl = clamp(y, 2, Ny - 1)
    n_trail = 0
    xmin = Nx
    xmax = 1
    @inbounds for x in 2:(Nx - 1)
        has = false
        for z in 2:(Nz - 1)
            n = lbm_n_t7(x, yl, z, Nx, Ny)
            su = fl[n] & TYPE_SU
            (su == TYPE_F || su == TYPE_I) || continue
            fsA[n] < 0.5f0 || continue
            has = true
            break
        end
        if has
            n_trail += 1
            xmin = min(xmin, x)
            xmax = max(xmax, x)
        end
    end
    return n_trail, xmin, xmax
end

@testset "short moving DED track: melt trail, powder, nuclei" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 32, 24, 32
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
        n = lbm_n_t7(x, y, z, Nx, Ny)
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
    x0 = 10.0f0
    x1 = 20.0f0
    yc = Float32(Ny + 1) / 2
    P = 180.0
    dx_noz = 3.0f0
    model.laser = Laser(units; P=P, w=2.0, x=x0, y=yc,
                        z=Float32(Nz) - 1.1f0, nrays=7, max_bounce=4, skin=12)
    model.powder_jet = PowderJet(units; mdot=5.0e-5, w=2.0, v=3.0,
                                 x=x0 - dx_noz, y=yc, z=Float32(Nz) - 1.2f0,
                                 dir=(0, 0, -1), nparcels=8, enabled=false,
                                 agent_frac=0.4)
    LatticeBoltzmann.initialize!(model)
    iy = (Ny + 1) ÷ 2
    n_hold = 240
    n_scan = 80
    with_logger(NullLogger()) do
        run!(model, n_hold)
    end
    depth0 = liquid_column_depth(model, round(Int, x0), iy)
    fl0 = Array(model.domains[1].flags.data)
    fs0 = Array(model.domains[1].fs.data)
    T0 = Array(model.domains[1].T.data)
    nF0 = count(n -> (fl0[n] & TYPE_SU) == TYPE_F, eachindex(fl0))
    nliq0 = count(n -> ((fl0[n] & TYPE_SU) == TYPE_F || (fl0[n] & TYPE_SU) == TYPE_I) &&
                       fs0[n] < 0.5f0, eachindex(fl0))
    @info "DED track after hold" depth0 nliq0 nF0 Tmax=maximum(T0) x=model.laser.x
    @test depth0 >= 8
    @test model.laser.x == x0
    @test model.laser.P == P

    mix_liquid_c_track!(model)
    model.laser.skin = 4
    model.bubbles.nucleation.enabled = true
    cores = nucleate_bubbles!(model, model.domains[1]; force=true)
    update_bubbles!(model)
    @test !isempty(cores)
    model.powder_jet.enabled = true

    v_lat = (x1 - x0) / Float32(n_scan)
    with_logger(NullLogger()) do
        for k in 1:n_scan
            x = x0 + v_lat * Float32(k)
            set_laser_position!(model.laser, x, yc)
            set_powder_jet_position!(model.powder_jet, x - dx_noz, yc)
            aim_powder_jet!(model.powder_jet, x, yc, Float32(Hfill))
            run!(model, 1)
        end
    end

    d = model.domains[1]
    fl = Array(d.flags.data)
    uA = Array(d.u.data)
    TA = Array(d.T.data)
    aA = Array(d.a.data)
    resA = Array(d.a_res.data)
    fsA = Array(d.fs.data)
    cA = Array(d.c.data)
    nF = 0
    umax = 0.0f0
    Tmax = -Inf32
    nliq = 0
    c_liq = 0.0f0
    @inbounds for n in eachindex(fl)
        su = fl[n] & TYPE_SU
        su == TYPE_F && (nF += 1)
        (su == TYPE_F || su == TYPE_I) || continue
        umax = max(umax, hypot(uA[n, 1], uA[n, 2], uA[n, 3]))
        Tmax = max(Tmax, TA[n])
        if fsA[n] < 0.5f0
            nliq += 1
            c_liq = max(c_liq, cA[n])
        end
    end
    depth_start = liquid_column_depth(model, round(Int, x0), iy)
    depth_end = liquid_column_depth(model, round(Int, model.laser.x), iy)
    n_trail, xmin, xmax = trail_span(model, iy)
    fm = foam_metrics(model)
    nfar = 4 + 3 * Nx + 3 * Nx * Ny
    zs = Int[]
    @inbounds for i in eachindex(model.bubbles.label)
        model.bubbles.label[i] > 0 || continue
        push!(zs, (i - 1) ÷ (Nx * Ny) + 1)
    end
    zbar = isempty(zs) ? 0.0 : sum(zs) / length(zs)
    @info "DED track after scan" model.laser.x model.powder_jet.x depth_start depth_end n_trail xmin xmax nliq a=sum(aA) res=sum(resA) c_liq fm.n_planted fm.nb zbar umax Tmax nF0 nF P=model.laser.P skin=model.laser.skin
    @test model.laser.x ≈ x1 atol=0.2
    @test model.laser.x > x0 + 8
    @test model.powder_jet.x < model.laser.x - 1
    @test model.laser.P == P
    @test depth_start >= 4
    @test depth_end >= 1
    @test n_trail >= 6
    @test xmax - xmin >= 6
    @test xmin <= round(Int, x0) + 2
    @test xmax >= round(Int, (x0 + x1) / 2)
    @test sum(aA) + sum(resA) > 0.05
    @test aA[nfar] < 1.0f-4
    @test c_liq > 1.05f0
    @test fm.n_planted >= 1
    @test fm.nb >= 1
    @test zbar > 4 && zbar <= Hfill + 1
    @test umax < 0.50f0
    @test nF > 0.85 * nF0
    @test isfinite(Tmax) && Tmax < 8
end
