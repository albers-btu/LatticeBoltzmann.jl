using Test
using LatticeBoltzmann
using KernelAbstractions
using Logging

@inline function lbm_n_c(x, y, z, Nx, Ny)
    return x + (y - 1) * Nx + (z - 1) * Nx * Ny
end

function crucible_model(Nx, Ny, Nz, Hfill; a0=0.10f0, k_a=0.08f0, Tlat=1.15f0,
                        k_H=3.0f0, α_c=0.2f0, c_star=1.05f0, d_min=6, n_max=8,
                        every=5, n_over=1.3f0, Λ=0.4f0, K0=1.0f-3)
    nuc = Nucleation{Float32}(; d_min=d_min, R=1, c_star=c_star, p_cell=1,
                              n_max=n_max, n_over=n_over, every=every, n_total_max=32)
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, α_c=α_c, k_H=k_H, fz=0, σ=0,
                  k_a=k_a, E_a=0, Y_a=1, a_fs_max=1, T_avg=Tlat,
                  Λ=Λ, Ts=1.0f0, Tl=1.0f0, K0=K0,
                  backend=CPU(), workgroup=64,
                  bubbles=BubbleTracker{Float32}(; nucleation=nuc))
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Tlat, Nx * Ny * Nz)
    ah = zeros(Float32, Nx * Ny * Nz)
    ch = fill(k_H * P_ATM_LAT, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_c(x, y, z, Nx, Ny)
        if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
            host[n] = TYPE_S
            Th[n] = Tlat
        elseif z <= Hfill
            host[n] = TYPE_F
            ah[n] = a0
        else
            host[n] = TYPE_G
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].a.data, ah)
    copyto!(model.domains[1].c.data, ch)
    return model
end

@testset "crucible foams: nuclei and porosity" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz, Hfill = 20, 20, 24, 14
    model = crucible_model(Nx, Ny, Nz, Hfill)
    LatticeBoltzmann.initialize!(model)
    m0 = foam_metrics(model)
    @test m0.nb == 0
    @test m0.porosity == 0
    h0 = m0.fill_z
    inv0 = m0.agent_total
    with_logger(NullLogger()) do
        run!(model, 25)
    end
    m1 = foam_metrics(model)
    @info "crucible foam" m0_fill=h0 m1...
    @test m1.n_planted >= 1
    @test m1.nb >= 1
    @test m1.V_bubble >= 1
    @test m1.porosity > 0
    @test m1.agent_total ≈ inv0 rtol=0.12
    @test m1.fill_z > 0
end

@testset "freeze traps enclosed gas and blocks nucleation" begin
    @test SURFACE
    Nx, Ny, Nz, Hfill = 20, 20, 24, 14
    model = crucible_model(Nx, Ny, Nz, Hfill; every=5, n_max=6)
    LatticeBoltzmann.initialize!(model)
    with_logger(NullLogger()) do
        run!(model, 20)
    end
    m_hot = foam_metrics(model)
    @test m_hot.nb >= 1
    planted_hot = m_hot.n_planted
    fsA = ones(Float32, Nx * Ny * Nz)
    copyto!(model.domains[1].fs.data, fsA)
    n_new = nucleate_bubbles!(model, model.domains[1]; force=true)
    @test isempty(n_new)
    @test model.bubbles.nucleation.n_planted == planted_hot
    with_logger(NullLogger()) do
        run!(model, 12)
    end
    m_cold = foam_metrics(model)
    @info "freeze trap" nb_hot=m_hot.nb V_hot=m_hot.V_bubble nb_cold=m_cold.nb V_cold=m_cold.V_bubble n_frozen=m_cold.n_frozen
    @test m_cold.nb >= 1
    @test m_cold.V_bubble >= 1
    @test m_cold.n_frozen > 0
end

function mean_a_band(model, zlo, zhi)
    d = model.domains[1]
    Nx, Ny, Nz = Int(d.Nx), Int(d.Ny), Int(d.Nz)
    flags = Array(d.flags.data)
    aA = Array(d.a.data)
    s = 0.0
    cnt = 0
    for z in zlo:zhi, y in 2:(Ny - 1), x in 2:(Nx - 1)
        n = lbm_n_c(x, y, z, Nx, Ny)
        (flags[n] & TYPE_SU) == TYPE_F || continue
        s += Float64(aA[n])
        cnt += 1
    end
    return cnt > 0 ? s / cnt : 0.0
end

@testset "hot bottom burns agent first" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz, Hfill = 16, 16, 22, 14
    T_bot, T_bulk = 1.4f0, 1.02f0
    nuc = Nucleation{Float32}(; enabled=false)
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, α_c=0.2f0, k_H=3.0f0,
                  fz=-1.0f-5, σ=0.005f0, k_a=1.5f0, E_a=6.0f0, Y_a=1,
                  a_fs_max=0.5f0, T_avg=T_bulk, Λ=0.5f0, Ts=1.0f0, Tl=1.0f0,
                  K0=1.0f-3, backend=CPU(), workgroup=64,
                  bubbles=BubbleTracker{Float32}(; nucleation=nuc))
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(T_bulk, Nx * Ny * Nz)
    ah = zeros(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_c(x, y, z, Nx, Ny)
        if z == 1
            host[n] = TYPE_S | TYPE_T
            Th[n] = T_bot
        elseif x == 1 || x == Nx || y == 1 || y == Ny || z == Nz
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
            ah[n] = 0.08f0
        else
            host[n] = TYPE_G
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].a.data, ah)
    fill!(model.domains[1].c.data, 3.0f0 * P_ATM_LAT)
    LatticeBoltzmann.initialize!(model)
    with_logger(NullLogger()) do
        run!(model, 80)
    end
    a_bot = mean_a_band(model, 2, 3)
    a_top = mean_a_band(model, Hfill - 2, Hfill - 1)
    @info "hot-bottom agent" a_bot a_top
    @test a_bot < a_top * 0.9
end

@testset "cold Dirichlet walls freeze from the boundary" begin
    @test SURFACE
    Nx, Ny, Nz, Hfill = 16, 16, 20, 12
    model = crucible_model(Nx, Ny, Nz, Hfill; every=4, n_max=4, k_a=0.08f0,
                           a0=0.10f0, Tlat=1.15f0)
    LatticeBoltzmann.initialize!(model)
    with_logger(NullLogger()) do
        run!(model, 16)
    end
    m_hot = foam_metrics(model)
    d = model.domains[1]
    host = Array(d.flags.data)
    Th = Array(d.T.data)
    T_cold = 0.65f0
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n_c(x, y, z, Nx, Ny)
        if z == 1 || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S | TYPE_T
            Th[n] = T_cold
        end
    end
    copyto!(d.flags.data, host)
    copyto!(d.T.data, Th)
    with_logger(NullLogger()) do
        run!(model, 40)
    end
    m_cold = foam_metrics(model)
    @info "wall freeze" frozen_hot=m_hot.n_frozen frozen_cold=m_cold.n_frozen nb=m_cold.nb
    @test m_cold.n_frozen > m_hot.n_frozen
    @test m_cold.nb >= 1 || m_hot.nb == 0
end
