using Test
using LatticeBoltzmann
using KernelAbstractions
using Logging

@inline function lbm_n_a(x, y, z, Nx, Ny)
    return x + (y - 1) * Nx + (z - 1) * Nx * Ny
end

function agent_box(Nx, Ny, Nz; Tlat=1.0f0, a0=0.02f0, k_a=0.02f0, E_a=0.0f0,
                   Y_a=1.0f0, k_H=3.0f0, α_c=0.0f0, fs_max=1.0f0)
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, α_c=α_c, k_H=k_H, fz=0, σ=0,
                  k_a=k_a, E_a=E_a, Y_a=Y_a, a_fs_max=fs_max, T_avg=Tlat,
                  backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Tlat, Nx * Ny * Nz)
    ah = zeros(Float32, Nx * Ny * Nz)
    nliq = 0
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_a(x, y, z, Nx, Ny)
        if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
            host[n] = TYPE_S
        else
            host[n] = TYPE_F
            ah[n] = a0
            nliq += 1
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].a.data, ah)
    return model, nliq
end

@testset "isothermal TGA a(t) = a0 exp(-k t)" begin
    @test SURFACE && TEMPERATURE
    a0 = 0.05f0
    k_a = 0.03f0
    model, nliq = agent_box(12, 8, 12; a0=a0, k_a=k_a, E_a=0, Tlat=1.0f0)
    LatticeBoltzmann.initialize!(model)
    inv0 = agent_inventory(model.domains[1])
    nsteps = 40
    with_logger(NullLogger()) do
        run!(model, nsteps)
    end
    inv1 = agent_inventory(model.domains[1])
    a_an = a0 * exp(-k_a * nsteps) * nliq
    @info "TGA" a0=inv0.a a=inv1.a a_an res=inv1.res dissolved=inv1.dissolved nliq
    @test inv1.a ≈ a_an rtol=0.02
    @test inv1.total ≈ inv0.total rtol=0.01
    @test inv1.dissolved ≈ (inv0.a - inv1.a) rtol=0.05
end

@testset "Arrhenius is slower when E/T is larger" begin
    @test SURFACE
    a0 = 0.04f0
    k_a = 0.2f0
    hot, _ = agent_box(10, 8, 10; a0=a0, k_a=k_a, E_a=0.0f0, Tlat=1.0f0)
    cold, _ = agent_box(10, 8, 10; a0=a0, k_a=k_a, E_a=4.0f0, Tlat=1.0f0)
    LatticeBoltzmann.initialize!(hot)
    LatticeBoltzmann.initialize!(cold)
    with_logger(NullLogger()) do
        run!(hot, 20)
        run!(cold, 20)
    end
    ih = agent_inventory(hot.domains[1])
    ic = agent_inventory(cold.domains[1])
    @info "Arrhenius T" a_hot=ih.a a_cold=ic.a k_hot=arrhenius_k(k_a, 0.0f0, 1.0f0) k_cold=arrhenius_k(k_a, 4.0f0, 1.0f0)
    @test ic.a > ih.a * 1.5f0
    @test ih.res > ic.res
end

@testset "fs_max skips solid cells" begin
    @test SURFACE
    model, nliq = agent_box(10, 8, 10; a0=0.03f0, k_a=0.2f0, E_a=0, fs_max=0.5f0)
    fsh = fill(1.0f0, 10 * 8 * 10)
    copyto!(model.domains[1].fs.data, fsh)
    LatticeBoltzmann.initialize!(model)
    inv0 = agent_inventory(model.domains[1])
    with_logger(NullLogger()) do
        run!(model, 15)
    end
    inv1 = agent_inventory(model.domains[1])
    @test inv1.a ≈ inv0.a rtol=0.02
    @test inv1.res < 1.0f-4 * inv0.a
end

@testset "powder agent_frac lands in a" begin
    @test SURFACE
    Nx, Ny, Nz = 12, 12, 16
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, τ_p=0,
                  backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    Hfill = 8
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_a(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
        else
            host[n] = TYPE_G
        end
    end
    copyto!(model.domains[1].flags.data, host)
    model.powder_jet = PowderJet{Float32}(; mdot=0, w=2, v=8, x=6, y=6, z=Float32(Hfill) + 2.5f0,
                                          nparcels=1, agent_frac=0.4f0, enabled=true)
    LatticeBoltzmann.initialize!(model)
    J = model.powder_jet
    J.px[1] = 6; J.py[1] = 6; J.pz[1] = Float32(Hfill) + 2.5f0
    J.pvx[1] = 0; J.pvy[1] = 0; J.pvz[1] = -J.v
    J.pm[1] = 2.0f0
    J.alive[1] = true
    LatticeBoltzmann.advance_powder_jet!(model, model.domains[1])
    aA = Array(model.domains[1].a.data)
    Sa = sum(aA)
    @info "powder agent" Sa
    @test Sa ≈ 0.4f0 * 2.0f0 atol=0.05
end

@testset "powder agent_frac lands in a when τ_p > 0" begin
    @test SURFACE
    Nx, Ny, Nz = 12, 12, 16
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, τ_p=10.0f0,
                  backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    Hfill = 8
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_a(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
        else
            host[n] = TYPE_G
        end
    end
    copyto!(model.domains[1].flags.data, host)
    model.powder_jet = PowderJet{Float32}(; mdot=0, w=2, v=8, x=6, y=6,
                                          z=Float32(Hfill) + 2.5f0,
                                          nparcels=1, agent_frac=0.4f0, enabled=true)
    LatticeBoltzmann.initialize!(model)
    J = model.powder_jet
    J.px[1] = 6; J.py[1] = 6; J.pz[1] = Float32(Hfill) + 2.5f0
    J.pvx[1] = 0; J.pvy[1] = 0; J.pvz[1] = -J.v
    J.pm[1] = 2.0f0
    J.alive[1] = true
    LatticeBoltzmann.advance_powder_jet!(model, model.domains[1])
    aA = Array(model.domains[1].a.data)
    mpA = Array(model.domains[1].mp.data)
    @test sum(aA) ≈ 0.4f0 * 2.0f0 atol=0.05
    @test sum(mpA) ≈ 0.6f0 * 2.0f0 atol=0.05
end
