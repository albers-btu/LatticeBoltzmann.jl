using Test
using LatticeBoltzmann
using KernelAbstractions
using Logging

@inline lbm_n_a5(x, y, z, Nx, Ny) = x + (y - 1) * Nx + (z - 1) * Nx * Ny

function agent_pad_model(; Nx=24, Ny=24, Nz=32, Hfill=20, a0=0.08f0)
    @test SURFACE && TEMPERATURE
    si_H = 0.002
    Lpad = max(4, Hfill - 2)
    m = si_H / Lpad
    st = 0.1 * m^2 / 7.5e-6
    units = Units(Lpad, 0.05, 1, si_H, 0.05 * m / st, 8000.0; T=Float32,
                  K=1673.0f0, cp=500.0f0)
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, α_c=0, k_H=3.0f0, fz=0, σ=0.02f0,
                  k_a=0.08f0, E_a=4.0f0, Y_a=1, a_fs_max=0.5f0,
                  Λ=0.45f0, Ts=1.0f0, Tl=1.0f0, K0=1.0f-3, T_avg=0.97f0,
                  backend=CPU(), workgroup=64)
    model.units = units
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(0.97f0, Nx * Ny * Nz)
    fsh = ones(Float32, Nx * Ny * Nz)
    ah = zeros(Float32, Nx * Ny * Nz)
    ch = fill(3.0f0 * P_ATM_LAT, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_a5(x, y, z, Nx, Ny)
        if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
            host[n] = TYPE_S
            fsh[n] = 1
        elseif z <= Hfill
            host[n] = TYPE_F
            ah[n] = a0
        else
            host[n] = TYPE_G
            fsh[n] = 0
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].fs.data, fsh)
    copyto!(model.domains[1].a.data, ah)
    copyto!(model.domains[1].c.data, ch)
    LatticeBoltzmann.initialize!(model)
    return model, a0
end

function agent_spot_stats(model, a0)
    d = model.domains[1]
    fl = Array(d.flags.data)
    fsA = Array(d.fs.data)
    aA = Array(d.a.data)
    cA = Array(d.c.data)
    nliq = nsol = 0
    a_liq = a_sol = 0.0
    c_liq_max = c_sol_max = 0.0f0
    n_hi_c_solid = 0
    @inbounds for n in eachindex(fl)
        su = fl[n] & TYPE_SU
        (su == TYPE_F || su == TYPE_I) || continue
        if fsA[n] < 0.5f0
            nliq += 1
            a_liq += Float64(aA[n])
            c_liq_max = max(c_liq_max, cA[n])
        elseif fsA[n] >= 0.99f0
            nsol += 1
            a_sol += Float64(aA[n])
            c_sol_max = max(c_sol_max, cA[n])
            # Never-melted solid: a still a0. High c here would mean dump in solid.
            aA[n] > 0.95f0 * a0 && cA[n] > 1.05f0 && (n_hi_c_solid += 1)
        end
    end
    xc = (Int(d.Nx) + 1) ÷ 2
    yc = (Int(d.Ny) + 1) ÷ 2
    depth = liquid_column_depth(model, xc, yc)
    return (; nliq, nsol, depth,
            a_liq = nliq == 0 ? 0.0 : a_liq / nliq,
            a_sol = nsol == 0 ? 0.0 : a_sol / nsol,
            c_liq_max, c_sol_max, n_hi_c_solid, a0)
end

@testset "Arrhenius dumps a→c in laser liquid only, solid a unchanged" begin
    Nx, Ny, Nz = 24, 24, 32
    Hfill = 20
    a0 = 0.08f0
    model, _ = agent_pad_model(; Nx, Ny, Nz, Hfill, a0)
    xc = Float32(Nx + 1) / 2
    yc = Float32(Ny + 1) / 2
    model.laser = Laser(model.units; P=180.0, w=2.0, x=xc, y=yc,
                        z=Float32(Nz) - 1.1f0, nrays=7, max_bounce=4, skin=12)
    with_logger(NullLogger()) do
        run!(model, 240)
    end
    st = agent_spot_stats(model, a0)
    @info "laser agent" st
    @test st.depth >= 12
    @test st.nliq > 100
    @test st.a_sol > 0.90 * a0
    @test st.a_liq < 0.90 * a0
    @test st.c_liq_max > st.c_sol_max
    @test st.c_liq_max > 1.05f0
    @test st.n_hi_c_solid == 0
    @test model.bubbles === nothing
end
