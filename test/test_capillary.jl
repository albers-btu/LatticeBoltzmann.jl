using Test
using LatticeBoltzmann
using KernelAbstractions
using Unitful
using Logging

@testset "capillary Δt sets SI σ to σ_lat" begin
    si_H = 1.0e-3u"m"
    si_ρ = 8000u"kg/m^3"
    si_σ = 1.5u"N/m"
    L = 20
    U = Units(si_H, si_ρ, si_σ; x=L, σ_lat=0.03, T=Float32)
    @test lbm_σ(U, si_σ) ≈ 0.03 rtol=1e-5
    @test U.s ≈ capillary_s(U.m, 8000, 1.5; σ_lat=0.03)
    p_atm_k = lbm_p(U, 101325)
    R = 5
    Δp_si = 2 * 1.5 / (R * U.m)
    @test lbm_p(U, Δp_si) ≈ 2 * lbm_σ(U, si_σ) / R rtol=1e-4
    @test lbm_p(U, Δp_si) / p_atm_k ≈ Δp_si / 101325 rtol=1e-4
    @test p_atm_k < 0.2f0
    g_lat = lbm_g(U, 9.81)
    Bo_si = 8000 * 9.81 * (1e-3)^2 / 1.5
    Bo_lat = g_lat * L^2 / lbm_σ(U, si_σ)
    @test Bo_lat ≈ Bo_si rtol=1e-4
end

@testset "SI Young–Laplace bubble holds radius" begin
    @test SURFACE
    si_H = 8.0e-4u"m"
    L = 16
    si_ρ = 8000u"kg/m^3"
    si_σ = 1.5u"N/m"
    U = Units(si_H, si_ρ, si_σ; x=L, σ_lat=0.03, T=Float32, K=1673, cp=500)
    m = U.m
    s = U.s
    ν_si = (0.1 * m^2 / s) * u"m^2/s"
    Nx = Ny = 20
    Nz = 24
    σlat = Float32(lbm_σ(U, si_σ))
    model = Model(Nx, Ny, Nz, U; ν=ν_si, α=2 * ν_si, σ=si_σ, gz=0,
                  backend=CPU(), workgroup=64,
                  bubbles=BubbleTracker{Float32}())
    @test model.domains[1].σ ≈ σlat rtol=1e-4
    xc = yc = (Nx + 1) / 2
    zc = 12.0f0
    R0 = 4.0f0
    host = zeros(UInt8, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
            host[n] = TYPE_S
        else
            dx, dy, dz = Float32(x) - xc, Float32(y) - yc, Float32(z) - zc
            host[n] = (dx * dx + dy * dy + dz * dz <= R0 * R0) ? TYPE_G : TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    LatticeBoltzmann.initialize!(model)
    b0 = bubble_records(model)[1]
    pL = young_laplace_p(σlat, b0.R, P_ATM_LAT)
    set_bubble_n!(model, b0.id, pL * b0.V)
    @test bubble_records(model)[1].p ≈ pL rtol=0.05
    # Kinetic 2σ/R equals the SI Laplace pressure converted with lbm_p.
    @test (pL - P_ATM_LAT) ≈ lbm_p(U, 2 * 1.5 / (b0.R * m)) rtol=0.15
    with_logger(NullLogger()) do
        run!(model, 16)
    end
    b = bubble_records(model)[1]
    @info "SI Laplace bubble" R0=b0.R R=b.R p=b.p pL σlat p_atm_k=lbm_p(U, 101325)
    @test b.R ≈ b0.R rtol=0.25
end

@testset "SI keyhole: T capped near Tv, Ma stays finite" begin
    @test SURFACE && TEMPERATURE
    si_H = 0.8e-3u"m"
    L = 16
    si_ρ = 8000u"kg/m^3"
    si_σ = 1.5u"N/m"
    si_Tm = 1673.0u"K"
    si_Tv = 3086.0u"K"
    U = Units(si_H, si_ρ, si_σ; x=L, σ_lat=0.03, T=Float32,
              K=ustrip(u"K", si_Tm), cp=500)
    m = U.m
    s = U.s
    α_si = 7.5e-6u"m^2/s"
    Nx, Ny, Nz = 16, 12, 24
    Hfill = 14
    Tm = Float32(lbm_T(U, si_Tm))
    model = Model(Nx, Ny, Nz, U;
                  ν=α_si, α=2 * α_si, σ=si_σ, gz=-9.81u"m/s^2",
                  latent=2.7e5u"J/kg", Ts=si_Tm, Tl=si_Tm, K0=1.0e-10u"m^2",
                  latent_v=7.45e6u"J/kg", T_v=si_Tv, M=0.0558u"kg/mol",
                  T_avg=Tm, backend=CPU(), workgroup=64)
    @test model.domains[1].σ ≈ 0.03f0 rtol=0.05
    @test model.domains[1].p0v ≈ lbm_p(U, 101325) rtol=1e-4
    @test abs(model.domains[1].fz) > 0
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Tm, Nx * Ny * Nz)
    fsh = zeros(Float32, Nx * Ny * Nz)
    Qh = zeros(Float32, Nx * Ny * Nz)
    xc, yc = Nx ÷ 2, Ny ÷ 2
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
            dx, dy = Float32(x - xc), Float32(y - yc)
            Qh[n] = 0.025f0 * exp(-(dx * dx + dy * dy) / 8.0f0)
        else
            host[n] = TYPE_G
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].fs.data, fsh)
    copyto!(model.domains[1].Q.data, Qh)
    LatticeBoltzmann.initialize!(model)
    fl = Array(model.domains[1].flags.data)
    Qh2 = Array(model.domains[1].Q.data)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        if (fl[n] & TYPE_SU) == TYPE_I
            dx, dy = Float32(x - xc), Float32(y - yc)
            Qh2[n] = 0.025f0 * exp(-(dx * dx + dy * dy) / 8.0f0)
        end
    end
    copyto!(model.domains[1].Q.data, Qh2)
    with_logger(NullLogger()) do
        run!(model, 80)
    end
    LatticeBoltzmann.moments!(model)
    TA = Array(model.domains[1].T.data)
    uA = Array(model.domains[1].u.data)
    fl = Array(model.domains[1].flags.data)
    Tmax = -Inf32
    umax = 0.0f0
    for n in eachindex(TA)
        su = fl[n] & TYPE_SU
        (su == TYPE_F || su == TYPE_I) || continue
        Tmax = max(Tmax, TA[n])
        umax = max(umax, hypot(uA[n, 1], uA[n, 2], uA[n, 3]))
    end
    Tv = Float32(lbm_T(U, si_Tv))
    @info "SI keyhole" Tmax Tv umax σ=model.domains[1].σ p0v=model.domains[1].p0v fz=model.domains[1].fz
    @test isfinite(Tmax) && isfinite(umax)
    @test Tmax > Tm
    @test Tmax < Tv + 0.5f0
    @test umax < 0.45f0
end
