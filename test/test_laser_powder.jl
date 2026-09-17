using Test
using LatticeBoltzmann
using KernelAbstractions
using Logging

@inline lbm_n_p6(x, y, z, Nx, Ny) = x + (y - 1) * Nx + (z - 1) * Nx * Ny

@testset "mixed powder + laser: a from the jet, dump only in the melt" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 24, 24, 32
    Hfill = 20
    si_H = 0.002
    Lpad = Hfill - 2
    m = si_H / Lpad
    st = 0.1 * m^2 / 7.5e-6
    units = Units(Lpad, 0.05, 1, si_H, 0.05 * m / st, 8000.0; T=Float32,
                  K=1673.0f0, cp=500.0f0)
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, α_c=0, k_H=3.0f0, fz=0, σ=0.02f0,
                  k_a=0.08f0, E_a=4.0f0, Y_a=1, a_fs_max=0.5f0,
                  Λ=0.45f0, Ts=1.0f0, Tl=1.0f0, K0=1.0f-3, T_avg=0.97f0,
                  τ_p=0, T_p=0.97f0,
                  backend=CPU(), workgroup=64)
    model.units = units
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(0.97f0, Nx * Ny * Nz)
    fsh = ones(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_p6(x, y, z, Nx, Ny)
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
    model.laser = Laser(units; P=180.0, w=2.0, x=xc, y=yc,
                        z=Float32(Nz) - 1.1f0, nrays=7, max_bounce=4, skin=12)
    model.powder_jet = PowderJet(units; mdot=5.0e-5, w=2.0, v=3.0,
                                 x=xc, y=yc, z=Float32(Nz) - 1.2f0,
                                 dir=(0, 0, -1), nparcels=8, agent_frac=0.4)
    LatticeBoltzmann.initialize!(model)
    @test sum(Array(model.domains[1].a.data)) == 0
    with_logger(NullLogger()) do
        run!(model, 240)
    end
    d = model.domains[1]
    aA = Array(d.a.data)
    resA = Array(d.a_res.data)
    cA = Array(d.c.data)
    fsA = Array(d.fs.data)
    fl = Array(d.flags.data)
    nfar = 4 + 3 * Nx + 3 * Nx * Ny
    a_far = aA[nfar]
    a_tot = sum(aA)
    res_tot = sum(resA)
    nliq = 0
    c_liq_max = 0.0f0
    a_liq = 0.0
    @inbounds for n in eachindex(fl)
        su = fl[n] & TYPE_SU
        (su == TYPE_F || su == TYPE_I) || continue
        fsA[n] < 0.5f0 || continue
        nliq += 1
        c_liq_max = max(c_liq_max, cA[n])
        a_liq += Float64(aA[n] + resA[n])
    end
    depth = liquid_column_depth(model, (Nx + 1) ÷ 2, (Ny + 1) ÷ 2)
    @info "mixed powder laser" a_tot res_tot a_far nliq c_liq_max depth
    @test a_tot + res_tot > 0.05
    @test a_far < 1.0f-4
    @test depth >= 8
    @test nliq > 100
    @test res_tot > 0
    @test c_liq_max > 1.0f0
    @test model.bubbles === nothing
end
