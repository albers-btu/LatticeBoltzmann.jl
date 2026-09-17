using Test
using LatticeBoltzmann
using KernelAbstractions
using Logging

@inline lbm_n_lo(x, y, z, Nx, Ny) = x + (y - 1) * Nx + (z - 1) * Nx * Ny

function opaque_pad(Nx, Ny, Nz, Hfill; with_I=false)
    @test SURFACE && TEMPERATURE
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, Λ=0, T_avg=1.0f0,
                  backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    phi = zeros(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_lo(x, y, z, Nx, Ny)
        if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
            host[n] = TYPE_S
        elseif with_I && z == Hfill
            host[n] = TYPE_I
            phi[n] = 0.0f0          # no PLIC plane → miss
        elseif z < Hfill || (!with_I && z <= Hfill)
            host[n] = TYPE_F
            phi[n] = 1.0f0
        else
            host[n] = TYPE_G
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].ϕ.data, phi)
    fill!(model.domains[1].Q.data, 0)
    return model
end

function q_stats(model)
    QA = Array(model.domains[1].Q.data)
    fl = Array(model.domains[1].flags.data)
    QI = QF = QG = 0.0f0
    nF = 0
    QmaxF = 0.0f0
    @inbounds for n in eachindex(QA)
        q = QA[n]
        su = fl[n] & TYPE_SU
        if su == TYPE_I
            QI += q
        elseif su == TYPE_F
            QF += q
            if q > 0
                nF += 1
                QmaxF = max(QmaxF, q)
            end
        elseif su == TYPE_G
            QG += q
        end
    end
    return (; QI, QF, QG, nF, QmaxF, Qtot = QI + QF)
end

@testset "TYPE_F deposit is spread over nskin, not one cell" begin
    Nx, Ny, Nz = 12, 12, 20
    Hfill = 12
    skin = 4
    P = 100.0f0
    model = opaque_pad(Nx, Ny, Nz, Hfill; with_I=false)
    d = model.domains[1]
    xc, yc = Float32(Nx ÷ 2), Float32(Ny ÷ 2)
    LatticeBoltzmann._walk_laser_ray!(
        d.Q.data, d.flags.data, d.ϕ.data,
        xc, yc, Float32(Nz) - 1.1f0, 0.0f0, 0.0f0, -1.0f0, P,
        3.27f0, 4.48f0, 1, skin, 1.0f0,
        Nx, Ny, Nz)
    st = q_stats(model)
    @info "F-spread" st
    @test st.QG == 0
    @test st.Qtot ≈ P atol=1.0f-3
    @test st.nF >= skin - 1
    @test st.QmaxF < 0.55f0 * P          # not the whole ray in one F cell
    @test st.QmaxF <= (P / skin) * 1.05f0
end

@testset "PLIC miss on TYPE_I does not dump the full ray into TYPE_F" begin
    Nx, Ny, Nz = 12, 12, 20
    Hfill = 11
    skin = 4
    P = 100.0f0
    A0 = LatticeBoltzmann.fresnel_absorptance(1.0f0, 3.27f0, 4.48f0)
    model = opaque_pad(Nx, Ny, Nz, Hfill; with_I=true)
    d = model.domains[1]
    xc, yc = Float32(Nx ÷ 2), Float32(Ny ÷ 2)
    LatticeBoltzmann._walk_laser_ray!(
        d.Q.data, d.flags.data, d.ϕ.data,
        xc, yc, Float32(Nz) - 1.1f0, 0.0f0, 0.0f0, -1.0f0, P,
        3.27f0, 4.48f0, 4, skin, 1.0f0,
        Nx, Ny, Nz)
    st = q_stats(model)
    @info "I-miss opaque" st A0
    @test st.QG == 0
    @test st.Qtot > 0
    @test isfinite(st.Qtot)
    @test st.QmaxF < 0.55f0 * P          # leftover power not dumped in one F
    @test st.Qtot < 0.55f0 * P           # Fresnel, not P_left = P
    @test abs(st.Qtot - A0 * P) / P < 0.20f0
end
