using Test
using LatticeBoltzmann
using KernelAbstractions
using Logging

@inline lbm_n_c(x, y, z, Nx, Ny) = x + (y - 1) * Nx + (z - 1) * Nx * Ny

function cavity_units(Hfill)
    si_H = 0.002
    Lpad = max(4, Hfill - 2)
    m = si_H / Lpad
    st = 0.1 * m^2 / 7.5e-6
    return Units(Lpad, 0.05, 1, si_H, 0.05 * m / st, 8000.0; T=Float32, K=1673.0f0, cp=500.0f0)
end

function paint_open_dimple!(host, Nx, Ny, Nz, Hfill, xc, yc, r, depth)
    r2 = Float32(r * r)
    zbot = Hfill - depth
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_c(x, y, z, Nx, Ny)
        if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
            host[n] = TYPE_S
        elseif z > Hfill
            host[n] = TYPE_G
        elseif z >= zbot && (Float32(x) - xc)^2 + (Float32(y) - yc)^2 <= r2
            host[n] = TYPE_G
        else
            host[n] = TYPE_F
        end
    end
    return host
end

function cavity_model(; Nx=24, Ny=24, Nz=28, Hfill=16, r=2.5, depth=6, σ=0.02f0)
    @test SURFACE && TEMPERATURE
    units = cavity_units(Hfill)
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=σ, Λ=0.6f0,
                  Ts=1.0f0, Tl=1.0f0, K0=1.0f-3, T_avg=1.05f0,
                  backend=CPU(), workgroup=64)
    model.units = units
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(1.05f0, Nx * Ny * Nz)
    fsh = zeros(Float32, Nx * Ny * Nz)
    xc, yc = Float32(Nx + 1) / 2, Float32(Ny + 1) / 2
    paint_open_dimple!(host, Nx, Ny, Nz, Hfill, xc, yc, r, depth)
    for n in eachindex(host)
        if (host[n] & TYPE_BO) == TYPE_S
            fsh[n] = 1
            Th[n] = 1.05f0
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].fs.data, fsh)
    LatticeBoltzmann.initialize!(model)
    return model, xc, yc
end

function q_cavity_stats(model)
    QA = Array(model.domains[1].Q.data)
    fl = Array(model.domains[1].flags.data)
    QI = QF = QG = 0.0f0
    nQ = 0
    Qmax = 0.0f0
    imax = 1
    @inbounds for n in eachindex(QA)
        q = QA[n]
        su = fl[n] & TYPE_SU
        if q != 0 && (su == TYPE_F || su == TYPE_I)
            nQ += 1
            if q > Qmax
                Qmax = q
                imax = n
            end
        end
        su == TYPE_I && (QI += q)
        su == TYPE_F && (QF += q)
        su == TYPE_G && (QG += q)
    end
    return (; QI, QF, QG, nQ, Qmax, imax, Qtot = QI + QF)
end

function metal_flags_uT(model)
    d = model.domains[1]
    fl = Array(d.flags.data)
    TA = Array(d.T.data)
    uA = Array(d.u.data)
    nF = nI = 0
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
        umax = max(umax, hypot(uA[n, 1], uA[n, 2], uA[n, 3]))
    end
    return (; nF, nI, umax, Tmax, finite)
end

@testset "laser deposit in an open dimple is not a one-cell dump" begin
    Nx, Ny, Nz = 24, 24, 28
    Hfill = 16
    skin = 4
    P = 80.0
    model, xc, yc = cavity_model(; Nx, Ny, Nz, Hfill, r=2.5, depth=6, σ=0)
    units = model.units
    model.laser = Laser(units; P=P, w=2.5, x=xc, y=yc, z=Float32(Nz) - 1.1f0,
                        nrays=7, max_bounce=4, skin=skin)
    fill!(model.domains[1].Q.data, 0)
    LatticeBoltzmann.deposit_laser!(model, model.domains[1])
    st = q_cavity_stats(model)
    qfac = LatticeBoltzmann.laser_qfac(units)
    Pabs = st.Qtot / qfac
    A0 = fresnel_absorptance(1.0f0, 3.27f0, 4.48f0)
    fl = Array(model.domains[1].flags.data)
    su_max = fl[st.imax] & TYPE_SU
    @info "cavity deposit" st Pabs A0 su_max
    @test st.QG == 0
    @test st.Qtot > 0
    @test isfinite(st.Qtot) && isfinite(st.Qmax)
    @test st.nQ >= skin
    @test st.Qmax < 0.45f0 * st.Qtot     # not one cell of the absorbed power
    @test 0.15 * P < Pabs < 1.05 * P
    @test su_max == TYPE_I || su_max == TYPE_F
end

@testset "short run into a prescribed cavity does not collapse the pad" begin
    Nx, Ny, Nz = 24, 24, 28
    Hfill = 16
    model, xc, yc = cavity_model(; Nx, Ny, Nz, Hfill, r=2.5, depth=6, σ=0.02f0)
    s0 = metal_flags_uT(model)
    model.laser = Laser(model.units; P=40.0, w=2.5, x=xc, y=yc,
                        z=Float32(Nz) - 1.1f0, nrays=7, max_bounce=4, skin=4)
    with_logger(NullLogger()) do
        run!(model, 40)
    end
    s1 = metal_flags_uT(model)
    @info "cavity run" s0 s1
    @test s1.finite
    @test isfinite(s1.Tmax) && s1.Tmax < 20
    @test s1.umax < 0.30f0
    @test s1.nF > 0.85 * s0.nF
    @test s1.nI > 10
end
