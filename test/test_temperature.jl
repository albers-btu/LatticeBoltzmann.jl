using Test
using LatticeBoltzmann
using KernelAbstractions

@inline function lbm_n(x, y, z, Nx, Ny)
    return x + (y - 1) * Nx + (z - 1) * Nx * Ny
end

@inline function erf_as(x::Float32)
    t = 1.0f0 / (1.0f0 + 0.3275911f0 * abs(x))
    τ = t * (0.254829592f0 + t * (-0.284496736f0 + t * (1.421413741f0 +
        t * (-1.453152027f0 + t * 1.061405429f0))))
    y = 1.0f0 - τ * exp(-x * x)
    return ifelse(x >= 0, y, -y)
end

@inline erfc_as(x::Float32) = 1.0f0 - erf_as(x)

# Two-phase Neumann λ: X=2λ√(κ_l t),
# λ√π = Ste_l e^{-λ²}/erf(λ) - Ste_s √(κ_s/κ_l) e^{-λ² κ_l/κ_s}/erfc(λ√(κ_l/κ_s))
function neumann_lambda_two_phase(Ste_l::Float32, Ste_s::Float32, κs_over_κl::Float32)
    r = sqrt(κs_over_κl)
    invr = 1.0f0 / r
    lo, hi = 0.01f0, 2.0f0
    λ = 0.2f0
    for _ in 1:60
        λ = 0.5f0 * (lo + hi)
        t1 = Ste_l * exp(-λ * λ) / erf_as(λ)
        t2 = Ste_s * r * exp(-λ * λ * invr * invr) / erfc_as(λ * invr)
        f = t1 - t2 - λ * sqrt(Float32(π))
        f > 0 ? (lo = λ) : (hi = λ)
    end
    return λ
end

@testset "solid walls do not impose Tm" begin
    @test TEMPERATURE && SURFACE
    Nx, Ny, Nz = 12, 12, 16
    Hfill = 10
    Tcold = 0.2f0
    Tm = 1.0f0
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, Λ=0.2f0, Ts=Tm, Tl=Tm,
                  T_avg=Tm, backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx*Ny*Nz)
    Th = fill(Tcold, Nx*Ny*Nz)
    fsh = ones(Float32, Nx*Ny*Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z==1 || z==Nz || x==1 || x==Nx || y==1 || y==Ny
            host[n] = TYPE_S
        elseif z == 2
            host[n] = TYPE_F | TYPE_T
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].fs.data, fsh)
    LatticeBoltzmann.initialize!(model)
    run!(model, 80)
    LatticeBoltzmann.moments!(model)
    T = Array(model.domains[1].T.data)
    nbot = lbm_n(6, 6, 2, Nx, Ny)
    nedge = lbm_n(2, 2, 2, Nx, Ny)
    nwall = lbm_n(6, 6, 1, Nx, Ny)
    @info "cold walls" Tbot=T[nbot] Tedge=T[nedge] Twall=T[nwall]
    @test T[nbot] ≈ Tcold atol=0.02f0
    @test T[nedge] ≈ Tcold atol=0.02f0
    @test T[nwall] ≈ Tcold atol=0.02f0
end

@testset "TEMPERATURE Dirichlet conduction" begin
    @test TEMPERATURE
    Nx, Ny, Nz = 8, 8, 16
    α = 0.05f0
    model = Model(Nx, Ny, Nz, 0.05; α = α, β = 0.0f0, fz = 0.0f0, backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = ones(Float32, Nx * Ny * Nz)
    T_hot, T_cold = 1.5f0, 0.5f0
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz
            host[n] = TYPE_S
        elseif z == 2
            host[n] = TYPE_T
            Th[n] = T_hot
        elseif z == Nz - 1
            host[n] = TYPE_T
            Th[n] = T_cold
        else
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    LatticeBoltzmann.initialize!(model)
    run!(model, 800)
    LatticeBoltzmann.moments!(model)
    T = Array(model.domains[1].T.data)
    mid = lbm_n(4, 4, Nz ÷ 2, Nx, Ny)
    @test T[mid] > T_cold + 0.15f0
    @test T[mid] < T_hot - 0.15f0
    @test isfinite(T[mid])
end

@testset "TEMPERATURE volumetric Q heats uniformly" begin
    Nx = Ny = Nz = 8
    Qv = 2.0f-4
    nsteps = 40
    model = Model(Nx, Ny, Nz, 0.05; α = 0.05f0, β = 0.0f0, fz = 0.0f0, backend=CPU(), workgroup=64)
    fill!(model.domains[1].flags.data, TYPE_F)
    fill!(model.domains[1].Q.data, Qv)
    LatticeBoltzmann.initialize!(model)
    T0 = Array(model.domains[1].T.data)
    run!(model, nsteps)
    LatticeBoltzmann.moments!(model)
    T = Array(model.domains[1].T.data)
    ΔT = sum(T - T0) / length(T)
    @test isapprox(ΔT, nsteps * Qv; rtol=0.12)
    @test all(isfinite, T)
end

@testset "TEMPERATURE Neumann flux vs Fourier" begin
    Nx, Ny, Nz = 6, 6, 16
    α = 0.2f0
    q_in = 4.0f-3
    T_cold = 1.0f0
    model = Model(Nx, Ny, Nz, 0.2; α = α, β = 0.0f0, fz = 0.0f0, backend=CPU(), workgroup=64)
    k = thermal_k(model.domains[1])
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(T_cold, Nx * Ny * Nz)
    Qh = zeros(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz
            host[n] = TYPE_S
        elseif z == 2
            host[n] = TYPE_H
            Qh[n] = q_in
        elseif z == Nz - 1
            host[n] = TYPE_T
            Th[n] = T_cold
        else
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].Q.data, Qh)
    LatticeBoltzmann.initialize!(model)
    run!(model, 2500)
    LatticeBoltzmann.moments!(model)
    T = Array(model.domains[1].T.data)
    zH, zC = 2, Nz - 1
    L = Float32(zC - zH)
    nH = lbm_n(3, 3, zH, Nx, Ny)
    nC = lbm_n(3, 3, zC, Nx, Ny)
    nM = lbm_n(3, 3, (zH + zC) ÷ 2, Nx, Ny)
    T_H, T_M, T_C = T[nH], T[nM], T[nC]
    @test T_H > T_C + 0.05f0
    @test isapprox(T_H, T_cold + q_in / k * L; rtol=0.2)
    @test isapprox(T_M, T_cold + q_in / k * (L / 2); rtol=0.25)
    @test isfinite(T_H) && isfinite(T_M)
end

@testset "TEMPERATURE Robin large h approaches T∞" begin
    Nx, Ny, Nz = 6, 6, 12
    α = 0.2f0
    T∞, T_cold = 1.6f0, 0.8f0
    hconv = 8.0f0
    model = Model(Nx, Ny, Nz, 0.2; α = α, β = 0.0f0, fz = 0.0f0, backend=CPU(), workgroup=64)
    k = thermal_k(model.domains[1])
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Float32((T∞ + T_cold) / 2), Nx * Ny * Nz)
    hh = zeros(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz
            host[n] = TYPE_S
        elseif z == 2
            host[n] = TYPE_H
            Th[n] = T∞
            hh[n] = hconv
        elseif z == Nz - 1
            host[n] = TYPE_T
            Th[n] = T_cold
        else
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].h.data, hh)
    LatticeBoltzmann.initialize!(model)
    run!(model, 2000)
    LatticeBoltzmann.moments!(model)
    T = Array(model.domains[1].T.data)
    zH, zC = 2, Nz - 1
    L = Float32(zC - zH)
    Bi = hconv / k
    A = (T_cold + Bi * L * T∞) / (1 + Bi * L)
    nH = lbm_n(3, 3, zH, Nx, Ny)
    nF = lbm_n(3, 3, zH + 1, Nx, Ny)
    @test isapprox(T[nF], A + (T_cold - A) / L; rtol=0.2)
    @test T[nF] > T_cold
    @test T[nF] < T∞
end

@testset "SURFACE+T insulated liquid layer" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 8, 8, 20
    α = 0.2f0
    Qv = 2.0f-5
    T_b = 1.0f0
    Hfill = 12
    model = Model(Nx, Ny, Nz, 0.2; α = α, β = 0.0f0, fz = 0.0f0, σ = 0.0f0,
                  backend=CPU(), workgroup=64)
    k = thermal_k(model.domains[1])
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(T_b, Nx * Ny * Nz)
    Qh = zeros(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z == 2
            host[n] = TYPE_T
            Th[n] = T_b
            Qh[n] = Qv
        elseif z <= Hfill
            host[n] = TYPE_F
            Qh[n] = Qv
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].Q.data, Qh)
    LatticeBoltzmann.initialize!(model)
    run!(model, 4000)
    LatticeBoltzmann.moments!(model)
    T = Array(model.domains[1].T.data)
    flags = Array(model.domains[1].flags.data)
    zb = 2
    zmid = (zb + Hfill) ÷ 2
    nM = lbm_n(4, 4, zmid, Nx, Ny)
    nG = lbm_n(4, 4, Nz - 2, Nx, Ny)
    H = Float32(Hfill - zb)
    zstar = Float32(zmid - zb)
    Tan = T_b + (Qv / k) * (H * zstar - zstar^2 / 2)
    @test (flags[nM] & TYPE_F) == TYPE_F || (flags[nM] & TYPE_I) == TYPE_I
    @test (flags[nG] & TYPE_G) == TYPE_G
    @test T[nM] > T_b
    @test isapprox(T[nM], Tan; rtol=0.35)
    @test isfinite(T[nM])
end

@testset "Marangoni cavity: surface jet toward cold, σT sign flip" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 32, 4, 16
    ν = 0.1f0
    α = 0.2f0
    T_hot, T_cold = 1.5f0, 0.5f0
    ΔT = T_hot - T_cold
    σ0 = 0.02f0
    σT = -0.02f0
    Hfill = 12
    model = Model(Nx, Ny, Nz, ν; α = α, β = 0.0f0, fz = 0.0f0, σ = σ0, σT = σT,
                  T_avg = 1.0f0, backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = ones(Float32, Nx * Ny * Nz)
    xH, xC = 2, Nx - 1
    Lx = Float32(xC - xH)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx
            host[n] = TYPE_S
        elseif z <= Hfill
            if x == xH
                host[n] = TYPE_T
                Th[n] = T_hot
            elseif x == xC
                host[n] = TYPE_T
                Th[n] = T_cold
            else
                host[n] = TYPE_F
                Th[n] = T_hot + (T_cold - T_hot) * Float32(x - xH) / Lx
            end
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    LatticeBoltzmann.initialize!(model)
    run!(model, 6000)
    LatticeBoltzmann.moments!(model)
    u = Array(model.domains[1].u.data)
    flags = Array(model.domains[1].flags.data)
    ux_s = Float32[]
    ux_b = Float32[]
    zB = 4
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        su = flags[n] & TYPE_SU
        if su == TYPE_I && x > xH + 2 && x < xC - 2
            push!(ux_s, u[n, 1])
        elseif su == TYPE_F && z == zB && x > xH + 2 && x < xC - 2
            push!(ux_b, u[n, 1])
        end
    end
    @test !isempty(ux_s)
    us = sum(ux_s) / length(ux_s)
    ub = isempty(ux_b) ? 0.0f0 : sum(ux_b) / length(ux_b)
    H = Float32(Hfill - 1)
    u_est = abs(σT) * ΔT * H / (4 * ν * Lx)
    # σT < 0: surface pulled to cold (+x)
    @test us > 0.25f0 * u_est
    @test us < 3 * u_est
    @test ub < 0           # return flow
    @test isfinite(us)

    # sign flip
    model2 = Model(Nx, Ny, Nz, ν; α = α, β = 0.0f0, fz = 0.0f0, σ = σ0, σT = -σT,
                   T_avg = 1.0f0, backend=CPU(), workgroup=64)
    copyto!(model2.domains[1].flags.data, host)
    copyto!(model2.domains[1].T.data, Th)
    LatticeBoltzmann.initialize!(model2)
    run!(model2, 6000)
    LatticeBoltzmann.moments!(model2)
    u2 = Array(model2.domains[1].u.data)
    flags2 = Array(model2.domains[1].flags.data)
    ux_s2 = Float32[]
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if (flags2[n] & TYPE_SU) == TYPE_I && x > xH + 2 && x < xC - 2
            push!(ux_s2, u2[n, 1])
        end
    end
    us2 = sum(ux_s2) / length(ux_s2)
    @test us2 < 0
    @test us2 * us < 0
end

@testset "enthalpy invert and Darcy cap" begin
    Tm, Λ = 1.0f0, 0.75f0
    T, fl = LatticeBoltzmann.invert_enthalpy(Tm - 0.1f0, Tm, Tm, Λ)
    @test fl == 0
    @test T ≈ Tm - 0.1f0
    T, fl = LatticeBoltzmann.invert_enthalpy(Tm + 0.3f0, Tm, Tm, Λ)
    @test T ≈ Tm
    @test fl ≈ 0.3f0 / Λ
    T, fl = LatticeBoltzmann.invert_enthalpy(Tm + Λ + 0.2f0, Tm, Tm, Λ)
    @test fl == 1
    @test T ≈ Tm + 0.2f0
    Ts, Tl, Λm = 1.0f0, 1.1f0, 0.5f0
    T, fl = LatticeBoltzmann.invert_enthalpy(Ts, Ts, Tl, Λm)
    @test T ≈ Ts && fl ≈ 0
    T, fl = LatticeBoltzmann.invert_enthalpy(Tl + Λm, Ts, Tl, Λm)
    @test T ≈ Tl && isapprox(fl, 1; atol=1.0f-6)
    dx, dy, dz = LatticeBoltzmann.darcy_force(1.0f0, 0.1f0, 0.0f0, 0.0f0, 1.0f0, 0.1f0, 1.0f-3)
    @test dx ≈ -0.2f0
    @test dy == 0 && dz == 0
    dx, dy, dz = LatticeBoltzmann.darcy_force(0.0f0, 0.1f0, 0.0f0, 0.0f0, 1.0f0, 0.1f0, 1.0f-3)
    @test abs(dx) < 1.0f-5
    dx, dy, dz = LatticeBoltzmann.darcy_force(0.5f0, 0.0f0, 0.0f0, 0.0f0, 1.0f0, 0.1f0, 0.0f0)
    @test dx == 0 && dy == 0 && dz == 0
    @test LatticeBoltzmann.blend_phase(0.0f0, 0.4f0, 0.2f0) ≈ 0.2f0
    @test LatticeBoltzmann.blend_phase(1.0f0, 0.4f0, 0.2f0) ≈ 0.4f0
    @test LatticeBoltzmann.blend_phase(0.5f0, 0.4f0, 0.2f0) ≈ 0.3f0
    @test LatticeBoltzmann.prop_fs_T(0.0f0, 0.2f0, 0.0f0, 0.4f0, 0.1f0, 2.0f0, 1.0f0, 1.0f-6) ≈ 0.5f0
    @test LatticeBoltzmann.prop_fs_T(1.0f0, 0.2f0, 0.05f0, 0.4f0, 0.0f0, 3.0f0, 1.0f0, 1.0f-6) ≈ 0.3f0
    @test LatticeBoltzmann.prop_fs_T(0.0f0, 0.2f0, 0.0f0, 0.4f0, 0.1f0, 1.0f0, 1.0f0, 1.0f-6) ≈ 0.4f0
    pmin = LatticeBoltzmann.prop_fs_T(1.0f0, 0.2f0, 1.0f0, 0.4f0, 0.0f0, 0.0f0, 1.0f0, 0.05f0)
    @test pmin ≈ 0.05f0
    # linear law through 0 at cold T floors at 10% of the Tref value, not ~0
    pcold = LatticeBoltzmann.prop_fs_T(1.0f0, 0.2f0, 0.3f0, 0.4f0, 0.0f0, 0.2f0, 1.0f0, 1.0f-6)
    @test pcold ≈ 0.02f0
    @test LatticeBoltzmann.floor_prop(0.2f0, 1.0f-6) ≈ 0.02f0
    @test LatticeBoltzmann.omega_T_from_alpha(1.0f-6) <= 1.95f0
    @test LatticeBoltzmann.omega_from_nu(1.0f-8) <= 1.95f0
    @test LatticeBoltzmann.radiation_dT(2.0f0, 0.01f0, 1.0f0) ≈ 0.01f0 * (16.0f0 - 1.0f0)
    @test LatticeBoltzmann.radiation_dT(1.0f0, 0.01f0, 1.0f0) == 0
    @test LatticeBoltzmann.radiation_dT(2.0f0, 0.0f0, 1.0f0) == 0
    Qr = LatticeBoltzmann.radiation_dT(1.2f0, 10.0f0, 1.0f0)
    @test Qr ≈ 0.2f0 atol=1.0f-6
    ωT = LatticeBoltzmann.omega_T_from_alpha(0.2f0)
    @test isapprox(LatticeBoltzmann.thermal_conductivity(ωT), 0.1f0; rtol=1.0f-5)
end

@testset "surface radiation cools TYPE_I" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 12, 8, 16
    Hfill = 10
    Tinf, Thot = 1.0f0, 1.8f0
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, Λ=0.0f0,
                  T_avg=Tinf, C_rad=0.02f0, T_rad=Tinf,
                  backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx*Ny*Nz)
    Th = fill(Thot, Nx*Ny*Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z==1 || z==Nz || x==1 || x==Nx || y==1 || y==Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    LatticeBoltzmann.initialize!(model)
    fl = Array(model.domains[1].flags.data)
    T0a = Array(model.domains[1].T.data)
    s0 = 0.0f0; nI = 0
    for n in eachindex(fl)
        if (fl[n] & TYPE_SU) == TYPE_I
            s0 += T0a[n]; nI += 1
        end
    end
    @test nI > 0
    T0 = s0 / nI
    run!(model, 25)
    LatticeBoltzmann.moments!(model)
    T1a = Array(model.domains[1].T.data)
    s1 = 0.0f0
    for n in eachindex(fl)
        (fl[n] & TYPE_SU) == TYPE_I && (s1 += T1a[n])
    end
    T1 = s1 / nI
    @info "radiation TYPE_I" T0 T1 nI
    @test T1 < T0 - 0.02f0
    @test T1 > Tinf
end

# k(Tm)+kT*(Troom-Tm) < 0 used to clamp α to 1e-6, ω_T→2, T oscillates
# through Tm, pad melts, FSLBM converts the whole domain to gas.
@testset "cold pad with k(T) is not eaten" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 12, 10, 16
    Hfill = 8
    Tm, Tcold = 1.0f0, 0.18f0
    model = Model(Nx, Ny, Nz, 0.1f0;
                  α=0.2f0, α_s=0.1f0, α_l=0.2f0,
                  α_sT=0.145f0, α_lT=0.056f0,
                  ν_s=0.05f0, ν_l=0.1f0, ν_lT=-0.045f0,
                  fz=0, σ=0.01f0, σT=0, Tσ=Tm,
                  Λ=0.3f0, Ts=Tm, Tl=Tm, K0=0.02f0,
                  T_avg=Tm, backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Tcold, Nx * Ny * Nz)
    fsh = ones(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z == 2
            host[n] = TYPE_F | TYPE_T
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].fs.data, fsh)
    LatticeBoltzmann.initialize!(model)
    d = model.domains[1]
    fl0 = Array(d.flags.data)
    nF0 = count(n -> (fl0[n] & TYPE_SU) == TYPE_F, eachindex(fl0))
    mass0 = sum(Array(d.mass.data))
    run!(model, 80)
    LatticeBoltzmann.moments!(model)
    fl = Array(d.flags.data)
    TA = Array(d.T.data)
    nF = count(n -> (fl[n] & TYPE_SU) == TYPE_F, eachindex(fl))
    nI = count(n -> (fl[n] & TYPE_SU) == TYPE_I, eachindex(fl))
    mass1 = sum(Array(d.mass.data))
    Tmin = minimum(TA[i] for i in eachindex(fl) if (fl[i] & TYPE_SU) == TYPE_F || (fl[i] & TYPE_SU) == TYPE_I)
    Tmax = maximum(TA[i] for i in eachindex(fl) if (fl[i] & TYPE_SU) == TYPE_F || (fl[i] & TYPE_SU) == TYPE_I)
    @info "cold pad k(T)" nF0 nF nI mass0 mass1 Tmin Tmax
    @test nF == nF0
    @test nI > 0
    @test isapprox(mass1, mass0; rtol=0.02)
    @test Tmax < Tm
    @test all(isfinite, TA)
end

@testset "Enthalpy Stefan melting vs Neumann" begin
    @test TEMPERATURE
    Nx, Ny, Nz = 64, 4, 4
    ν = 0.1f0
    α = 0.2f0
    Tm = 1.0f0
    Tb = 1.15f0
    Λ = 0.75f0
    Ste = (Tb - Tm) / Λ
    model = Model(Nx, Ny, Nz, ν; α = α, β = 0.0f0, fz = 0.0f0, σ = 0.0f0,
                  Λ = Λ, Ts = Tm, Tl = Tm, K0 = 1.0f-3, T_avg = Tm,
                  backend=CPU(), workgroup=64)
    k = thermal_k(model.domains[1])
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Tm, Nx * Ny * Nz)
    fsh = ones(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
            host[n] = TYPE_S
        elseif x == 2
            host[n] = TYPE_T
            Th[n] = Tb
            fsh[n] = 0
        else
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].fs.data, fsh)
    LatticeBoltzmann.initialize!(model)
    nsteps = 4000
    run!(model, nsteps)
    LatticeBoltzmann.moments!(model)
    fsA = Array(model.domains[1].fs.data)
    uA = Array(model.domains[1].u.data)
    # Neumann λ: λ exp(λ²) erf(λ) = Ste/√π
    rhs = Ste / sqrt(Float32(π))
    lo, hi = 0.01f0, 2.0f0
    λ = 0.3f0
    for _ in 1:40
        λ = 0.5f0 * (lo + hi)
        f = λ * exp(λ^2) * erf_as(λ)
        f > rhs ? (hi = λ) : (lo = λ)
    end
    Xan = 2 * λ * sqrt(k * Float32(nsteps))
    y0, z0 = 2, 2
    xif = Nx - 1
    for x in 3:Nx-1
        n = lbm_n(x, y0, z0, Nx, Ny)
        if fsA[n] > 0.5f0
            xif = x
            break
        end
    end
    Xnum = Float32(xif - 2)
    nsolid = lbm_n(Nx - 3, y0, z0, Nx, Ny)
    @info "Stefan vs Neumann" Ste λ k Xnum Xan ratio=(Xnum / Xan)
    @test Xnum > 4
    @test isapprox(Xnum, Xan; rtol=0.35)
    @test hypot(uA[nsolid, 1], uA[nsolid, 2], uA[nsolid, 3]) < 0.01f0
    @test isfinite(Xnum)
end

@testset "two-phase Neumann melting k_s ≠ k_l" begin
    @test TEMPERATURE
    Nx, Ny, Nz = 64, 4, 4
    ν_l, ν_s = 0.1f0, 0.05f0
    α_l, α_s = 0.2f0, 0.4f0          # Model-α; κ = α/2
    Tm, Tb, Ti = 1.0f0, 1.15f0, 0.85f0
    Λ = 0.75f0
    Ste_l = (Tb - Tm) / Λ
    Ste_s = (Tm - Ti) / Λ
    model = Model(Nx, Ny, Nz, ν_l; α = α_l, α_s = α_s, α_l = α_l,
                  ν_s = ν_s, ν_l = ν_l, β = 0.0f0, fz = 0.0f0, σ = 0.0f0,
                  Λ = Λ, Ts = Tm, Tl = Tm, K0 = 1.0f-3, T_avg = Tm,
                  backend=CPU(), workgroup=64)
    k_l = thermal_k_l(model.domains[1])
    k_s = thermal_k_s(model.domains[1])
    @test k_l ≈ 0.1f0
    @test k_s ≈ 0.2f0
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Ti, Nx * Ny * Nz)
    fsh = ones(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
            host[n] = TYPE_S
        elseif x == 2
            host[n] = TYPE_T
            Th[n] = Tb
            fsh[n] = 0
        else
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].fs.data, fsh)
    LatticeBoltzmann.initialize!(model)
    nsteps = 4000
    run!(model, nsteps)
    LatticeBoltzmann.moments!(model)
    fsA = Array(model.domains[1].fs.data)
    uA = Array(model.domains[1].u.data)
    λ = neumann_lambda_two_phase(Ste_l, Ste_s, k_s / k_l)
    Xan = 2 * λ * sqrt(k_l * Float32(nsteps))
    y0, z0 = 2, 2
    xif = Nx - 1
    for x in 3:Nx-1
        n = lbm_n(x, y0, z0, Nx, Ny)
        if fsA[n] > 0.5f0
            xif = x
            break
        end
    end
    Xnum = Float32(xif - 2)
    nsolid = lbm_n(Nx - 3, y0, z0, Nx, Ny)
    nliq = lbm_n(4, y0, z0, Nx, Ny)
    @info "two-phase Neumann" Ste_l Ste_s k_l k_s λ Xnum Xan ratio=(Xnum / Xan)
    @test Xnum > 3
    @test isapprox(Xnum, Xan; rtol=0.40)
    @test hypot(uA[nsolid, 1], uA[nsolid, 2], uA[nsolid, 3]) < 0.01f0
    @test fsA[nliq] < 0.5f0
    @test isfinite(Xnum)
end

@testset "SURFACE × enthalpy: open-layer freeze vs Neumann" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 8, 8, 40
    ν = 0.1f0
    α = 0.2f0
    Tm = 1.0f0
    Tb = 0.85f0
    Λ = 0.75f0
    Ste = (Tm - Tb) / Λ
    Hfill = 32
    zB = 2
    model = Model(Nx, Ny, Nz, ν; α = α, β = 0.0f0, fz = 0.0f0, σ = 0.0f0,
                  Λ = Λ, Ts = Tm, Tl = Tm, K0 = 1.0f-3, T_avg = Tm,
                  backend=CPU(), workgroup=64)
    k = thermal_k(model.domains[1])
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Tm, Nx * Ny * Nz)
    fsh = zeros(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z == zB
            host[n] = TYPE_T
            Th[n] = Tb
            fsh[n] = 1
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].fs.data, fsh)
    LatticeBoltzmann.initialize!(model)
    nsteps = 4000
    run!(model, nsteps)
    LatticeBoltzmann.moments!(model)
    fsA = Array(model.domains[1].fs.data)
    uA = Array(model.domains[1].u.data)
    flags = Array(model.domains[1].flags.data)
    TA = Array(model.domains[1].T.data)
    rhs = Ste / sqrt(Float32(π))
    lo, hi = 0.01f0, 2.0f0
    λ = 0.3f0
    for _ in 1:40
        λ = 0.5f0 * (lo + hi)
        f = λ * exp(λ^2) * erf_as(λ)
        f > rhs ? (hi = λ) : (lo = λ)
    end
    Xan = 2 * λ * sqrt(k * Float32(nsteps))
    x0, y0 = 4, 4
    Xnum = 0.0f0
    for z in zB+1:Hfill-1
        n = lbm_n(x0, y0, z, Nx, Ny)
        np = lbm_n(x0, y0, z + 1, Nx, Ny)
        if fsA[n] >= 0.5f0 && fsA[np] < 0.5f0
            Xnum = Float32(z - zB) + (fsA[n] - 0.5f0) / (fsA[n] - fsA[np] + 1.0f-8)
            break
        end
    end
    zI = 0
    for z in 1:Nz
        n = lbm_n(x0, y0, z, Nx, Ny)
        if (flags[n] & TYPE_SU) == TYPE_I
            zI = z
        end
    end
    nG = lbm_n(x0, y0, Nz - 2, Nx, Ny)
    nsolid = lbm_n(x0, y0, zB + 2, Nx, Ny)
    @info "open-layer freeze vs Neumann" Ste λ k Xnum Xan ratio=(Xnum / Xan) zI Hfill
    @test Xnum > 4
    @test isapprox(Xnum, Xan; rtol=0.35)
    @test abs(zI - (Hfill + 1)) <= 2
    @test (flags[nG] & TYPE_G) == TYPE_G
    @test hypot(uA[nsolid, 1], uA[nsolid, 2], uA[nsolid, 3]) < 0.01f0
    @test isfinite(TA[nG])
    @test isfinite(Xnum)
end

@testset "Hertz–Knudsen evaporative dT" begin
    @test LatticeBoltzmann.evaporative_dT(1.0f0, 8.0f0, 1.5f0, 0.01f0, 0.5f0, 30.0f0) == 0
    @test LatticeBoltzmann.evaporative_dT(1.6f0, 0.0f0, 1.5f0, 0.01f0, 0.5f0, 30.0f0) == 0
    Qe = LatticeBoltzmann.evaporative_dT(1.6f0, 8.0f0, 1.5f0, 0.01f0, 0.5f0, 30.0f0)
    @test Qe > 0
    @test Qe ≤ 1.6f0 - 1.5f0 + 1.0f-6
    Qe2, mdot = LatticeBoltzmann.evaporative_flux(1.6f0, 8.0f0, 1.5f0, 0.01f0, 0.5f0, 30.0f0)
    @test Qe2 == Qe
    @test mdot ≈ Qe / 8.0f0
    @test LatticeBoltzmann.p_sat(1.5f0, 1.5f0, 1.0f0, 20.0f0) ≈ 1
    @test LatticeBoltzmann.p_sat(1.0f0, 1.5f0, 1.0f0, 20.0f0) < 0.05f0
end

@testset "surface evaporation caps T near Tv" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 16, 8, 20
    Hfill = 12
    Tm, Tv = 1.0f0, 1.5f0
    model = Model(Nx, Ny, Nz, 0.1f0; α = 0.2f0, β = 0.0f0, fz = 0.0f0, σ = 0.0f0,
                  Λ = 0.75f0, Ts = Tm, Tl = Tm, K0 = 1.0f-3, T_avg = Tm,
                  Λ_v = 8.0f0, T_v = Tv, C_hk = 0.08f0, p0v = 1.0f0, β_v = 20.0f0,
                  backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Tm, Nx * Ny * Nz)
    fsh = zeros(Float32, Nx * Ny * Nz)
    Qh = zeros(Float32, Nx * Ny * Nz)
    xc, yc = Nx ÷ 2, Ny ÷ 2
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
            fsh[n] = 0
            if z >= Hfill - 4
                dx, dy = Float32(x - xc), Float32(y - yc)
                Qh[n] = 0.03f0 * exp(-(dx * dx + dy * dy) / 8.0f0)
            end
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
        n = lbm_n(x, y, z, Nx, Ny)
        if (fl[n] & TYPE_SU) == TYPE_I
            dx, dy = Float32(x - xc), Float32(y - yc)
            Qh2[n] = 0.03f0 * exp(-(dx * dx + dy * dy) / 8.0f0)
        end
    end
    copyto!(model.domains[1].Q.data, Qh2)
    run!(model, 2500)
    LatticeBoltzmann.moments!(model)
    TA = Array(model.domains[1].T.data)
    fl = Array(model.domains[1].flags.data)
    Tmax_I = -Inf32
    Tmax_F = -Inf32
    for n in eachindex(TA)
        su = fl[n] & TYPE_SU
        su == TYPE_I && (Tmax_I = max(Tmax_I, TA[n]))
        su == TYPE_F && (Tmax_F = max(Tmax_F, TA[n]))
    end
    @info "evaporation cap" Tmax_I Tmax_F Tv
    @test isfinite(Tmax_I)
    @test Tmax_I > Tm
    @test Tmax_I < Tv + 0.40f0
end

@testset "recoil dimples free surface under a hot spot" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 16, 8, 20
    Hfill = 12
    Tm, Tv = 1.0f0, 1.5f0
    model = Model(Nx, Ny, Nz, 0.1f0; α = 0.2f0, β = 0.0f0, fz = 0.0f0, σ = 0.0f0,
                  Λ = 0.75f0, Ts = Tm, Tl = Tm, K0 = 1.0f-3, T_avg = Tm,
                  Λ_v = 8.0f0, T_v = Tv, C_hk = 0.08f0, p0v = 1.0f0, β_v = 20.0f0,
                  backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Tm, Nx * Ny * Nz)
    fsh = zeros(Float32, Nx * Ny * Nz)
    Qh = zeros(Float32, Nx * Ny * Nz)
    xc, yc = Nx ÷ 2, Ny ÷ 2
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
            fsh[n] = 0
            if z >= Hfill - 4
                dx, dy = Float32(x - xc), Float32(y - yc)
                Qh[n] = 0.03f0 * exp(-(dx * dx + dy * dy) / 8.0f0)
            end
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
        n = lbm_n(x, y, z, Nx, Ny)
        if (fl[n] & TYPE_SU) == TYPE_I
            dx, dy = Float32(x - xc), Float32(y - yc)
            Qh2[n] = 0.03f0 * exp(-(dx * dx + dy * dy) / 8.0f0)
        end
    end
    copyto!(model.domains[1].Q.data, Qh2)
    M0 = sum(Array(model.domains[1].mass.data))
    run!(model, 2500)
    LatticeBoltzmann.moments!(model)
    M1 = sum(Array(model.domains[1].mass.data))
    ϕA = Array(model.domains[1].ϕ.data)
    fl = Array(model.domains[1].flags.data)
    TA = Array(model.domains[1].T.data)
    function fill_z(x, y)
        ztop = 0.0f0
        for z in (Nz - 1):-1:2
            n = lbm_n(x, y, z, Nx, Ny)
            su = fl[n] & TYPE_SU
            if su == TYPE_I || (su == TYPE_F && ϕA[n] > 0.05f0)
                return Float32(z - 1) + ϕA[n]
            end
        end
        return ztop
    end
    hC = fill_z(xc, yc)
    hE = fill_z(3, yc)
    Tmax_I = -Inf32
    for n in eachindex(TA)
        (fl[n] & TYPE_SU) == TYPE_I && (Tmax_I = max(Tmax_I, TA[n]))
    end
    @info "recoil dimple" hC hE Tmax_I Tv M0 M1
    @test isfinite(hC) && isfinite(hE)
    @test hC < hE - 0.3f0
    @test Tmax_I < Tv + 0.40f0
    @test M0 > 10
    @test M1 < M0 - 0.5f0
end

@testset "mass source grows a bead" begin
    @test SURFACE
    Nx, Ny, Nz = 16, 8, 20
    Hfill = 12
    Tm = 1.0f0
    model = Model(Nx, Ny, Nz, 0.1f0; α = 0.2f0, β = 0.0f0, fz = 0.0f0, σ = 0.0f0,
                  Λ = 0.0f0, T_avg = Tm, backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Tm, Nx * Ny * Nz)
    fsh = zeros(Float32, Nx * Ny * Nz)
    Sh = zeros(Float32, Nx * Ny * Nz)
    xc, yc = Nx ÷ 2, Ny ÷ 2
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
            if z >= Hfill - 1
                dx, dy = Float32(x - xc), Float32(y - yc)
                Sh[n] = 0.008f0 * exp(-(dx * dx + dy * dy) / 8.0f0)
            end
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].fs.data, fsh)
    copyto!(model.domains[1].msrc.data, Sh)
    LatticeBoltzmann.initialize!(model)
    fl = Array(model.domains[1].flags.data)
    Sh2 = Array(model.domains[1].msrc.data)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if (fl[n] & TYPE_SU) == TYPE_I
            dx, dy = Float32(x - xc), Float32(y - yc)
            Sh2[n] = 0.008f0 * exp(-(dx * dx + dy * dy) / 8.0f0)
        end
    end
    copyto!(model.domains[1].msrc.data, Sh2)
    M0 = sum(Array(model.domains[1].mass.data))
    run!(model, 800)
    LatticeBoltzmann.moments!(model)
    M1 = sum(Array(model.domains[1].mass.data))
    ϕA = Array(model.domains[1].ϕ.data)
    fl = Array(model.domains[1].flags.data)
    function fill_z(x, y)
        for z in (Nz - 1):-1:2
            n = lbm_n(x, y, z, Nx, Ny)
            su = fl[n] & TYPE_SU
            if su == TYPE_I || (su == TYPE_F && ϕA[n] > 0.05f0)
                return Float32(z - 1) + ϕA[n]
            end
        end
        return 0.0f0
    end
    hC = fill_z(xc, yc)
    hE = fill_z(3, yc)
    @info "mass source bead" M0 M1 hC hE
    @test M1 > M0 + 1
    @test hC > hE + 0.3f0
end

@testset "unmelted powder decays; hot pool still captures" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 16, 8, 20
    Hfill = 12
    Tm = 1.0f0
    Tcold = 0.2f0
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, Λ=0.2f0,
                  Ts=Tm, Tl=Tm, T_avg=Tm, τ_p=8.0f0, T_p=Tcold,
                  backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx*Ny*Nz)
    Th = fill(Tcold, Nx*Ny*Nz)
    fsh = ones(Float32, Nx*Ny*Nz)
    Sh = zeros(Float32, Nx*Ny*Nz)
    xc, yc = Nx ÷ 2, Ny ÷ 2
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z==1 || z==Nz || x==1 || x==Nx || y==1 || y==Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].fs.data, fsh)
    LatticeBoltzmann.initialize!(model)
    fl = Array(model.domains[1].flags.data)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        su = fl[n] & TYPE_SU
        if su == TYPE_I || su == TYPE_G
            dx, dy = Float32(x - xc), Float32(y - yc)
            Sh[n] = 0.01f0 * exp(-(dx*dx + dy*dy) / 8.0f0)
        end
    end
    copyto!(model.domains[1].msrc.data, Sh)
    M0 = sum(Array(model.domains[1].mass.data))
    run!(model, 4)
    Mp = sum(Array(model.domains[1].mp.data))
    M1 = sum(Array(model.domains[1].mass.data))
    @info "cold powder" M0 M1 Mp
    @test Mp > 0.01f0
    @test abs(M1 - M0) < 0.05f0
    fill!(model.domains[1].msrc.data, 0)
    run!(model, 40)
    Mp2 = sum(Array(model.domains[1].mp.data))
    M2 = sum(Array(model.domains[1].mass.data))
    @info "cold powder after decay" Mp2 M2
    @test Mp2 < 0.1f0 * Mp
    @test abs(M2 - M0) < 0.05f0

    modelh = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, Λ=0.0f0,
                   T_avg=Tm, τ_p=20.0f0, T_p=Tm, backend=CPU(), workgroup=64)
    Th .= Tm
    fsh .= 0
    copyto!(modelh.domains[1].flags.data, host)
    copyto!(modelh.domains[1].T.data, Th)
    copyto!(modelh.domains[1].fs.data, fsh)
    LatticeBoltzmann.initialize!(modelh)
    fl = Array(modelh.domains[1].flags.data)
    fill!(Sh, 0)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if (fl[n] & TYPE_SU) == TYPE_I
            dx, dy = Float32(x - xc), Float32(y - yc)
            Sh[n] = 0.008f0 * exp(-(dx*dx + dy*dy) / 8.0f0)
        end
    end
    copyto!(modelh.domains[1].msrc.data, Sh)
    Mh0 = sum(Array(modelh.domains[1].mass.data))
    run!(modelh, 400)
    Mh1 = sum(Array(modelh.domains[1].mass.data))
    @info "hot powder capture" Mh0 Mh1
    @test Mh1 > Mh0 + 0.5f0
end

@testset "powder jet hits pad from above" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 24, 16, 24
    Hfill = 14
    Tm = 1.0f0
    Tcold = 0.2f0
    si_H = 0.002
    Lpad = Hfill - 2
    m = si_H / Lpad
    st = 0.1 * m^2 / 7.5e-6
    units = Units(Lpad, 0.05, 1, si_H, 0.05 * m / st, 8000.0; T=Float32, K=1673.0f0, cp=500.0f0)
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, Λ=0.2f0, Ts=Tm, Tl=Tm,
                  T_avg=Tm, τ_p=1.0f5, T_p=Tcold, backend=CPU(), workgroup=64)
    model.units = units
    host = zeros(UInt8, Nx*Ny*Nz)
    Th = fill(Tcold, Nx*Ny*Nz)
    fsh = ones(Float32, Nx*Ny*Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z==1 || z==Nz || x==1 || x==Nx || y==1 || y==Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].fs.data, fsh)
    model.powder_jet = PowderJet(units; mdot=1.0e-4, w=2.0, v=2.0,
                                 x=Nx/2, y=Ny/2, z=Float32(Nz)-1.2f0,
                                 dir=(0, 0, -1), nparcels=25)
    LatticeBoltzmann.initialize!(model)
    M0 = sum(Array(model.domains[1].mass.data))
    run!(model, 10)
    Mp = sum(Array(model.domains[1].mp.data))
    M1 = sum(Array(model.domains[1].mass.data))
    @info "powder jet" Mp M0 M1 nalive=count(model.powder_jet.alive)
    @test Mp > 0
    @test abs(M1 - M0) < 0.05f0
end

@testset "PLIC laser Fresnel deposit" begin
    A0 = LatticeBoltzmann.fresnel_absorptance(1.0f0, 3.27f0, 4.48f0)
    Ag = LatticeBoltzmann.fresnel_absorptance(0.0f0, 3.27f0, 4.48f0)
    @test 0.30f0 < A0 < 0.40f0
    @test Ag < 0.02f0
    @test TEMPERATURE && SURFACE
    Nx, Ny, Nz = 24, 16, 24
    Hfill = 14
    Tm = 1.0f0
    si_H = 0.002
    Lpad = Hfill - 2
    m = si_H / Lpad
    st = 0.1 * m^2 / 7.5e-6
    units = Units(Lpad, 0.05, 1, si_H, 0.05 * m / st, 8000.0; T=Float32, K=1673.0f0, cp=500.0f0)
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, Λ=0, T_avg=Tm,
                  backend=CPU(), workgroup=64)
    model.units = units
    host = zeros(UInt8, Nx*Ny*Nz)
    Th = fill(Tm, Nx*Ny*Nz)
    fsh = zeros(Float32, Nx*Ny*Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z==1 || z==Nz || x==1 || x==Nx || y==1 || y==Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].fs.data, fsh)
    LatticeBoltzmann.initialize!(model)
    las = Laser(units; P=200.0, w=2.0, x=Nx/2, y=Ny/2, z=Float32(Nz)-1.1f0,
                nrays=9, max_bounce=4, skin=1)
    model.laser = las
    fill!(model.domains[1].Q.data, 0)
    LatticeBoltzmann.deposit_laser!(model, model.domains[1])
    QA = Array(model.domains[1].Q.data)
    fl = Array(model.domains[1].flags.data)
    QI = 0.0f0
    QF = 0.0f0
    QG = 0.0f0
    for n in eachindex(QA)
        su = fl[n] & TYPE_SU
        su == TYPE_I && (QI += QA[n])
        su == TYPE_F && (QF += QA[n])
        su == TYPE_G && (QG += QA[n])
    end
    qfac = LatticeBoltzmann.laser_qfac(units)
    Pabs = (QI + QF) / qfac
    @info "laser deposit" QI QF QG A0 Pabs nray=length(las.Pray)
    @test QI + QF > 0
    @test isfinite(QI) && isfinite(QF)
    @test QG == 0
    @test QI > QF
    @test 0.20 * las.P < Pabs < 1.05 * las.P
    @test abs(Pabs - A0 * las.P) / las.P < 0.15
end

@testset "export VTK survives NaN/Inf and empty ρ" begin
    Nx, Ny, Nz = 8, 8, 8
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, Λ=0, T_avg=1.0f0,
                  backend=CPU(), workgroup=64)
    fill!(model.domains[1].flags.data, TYPE_F)
    model.domains[1].flags.data[1] = TYPE_S
    model.domains[1].T.data[2] = NaN32
    model.domains[1].T.data[3] = Inf32
    model.domains[1].ρ.data[4] = 0
    model.domains[1].mp.data[4] = 1.0f0
    model.domains[1].Q.data[5] = Inf32
    mktempdir() do d
        export!(model; dir=d)
        @test isfile(joinpath(d, "lbm.pvd"))
        @test isfile(joinpath(d, "lbm_00000000.vti"))
        pvd = read(joinpath(d, "lbm.pvd"), String)
        @test occursin("timestep=", pvd)
        @test !occursin("nan", lowercase(pvd))
        vti = read(joinpath(d, "lbm_00000000.vti"), String)
        @test occursin("Scalars=\"T\"", vti) || occursin("Scalars='T'", vti)
    end
    ρ = ones(Float32, 8)
    ρ[1] = 0
    ρ[2] = NaN32
    mp = Float32[1, 1, 0, 0, 0, 0, 0, 0]
    B = LatticeBoltzmann._vtk_fillfrac(mp, ρ, 2, 2, 2)
    @test all(isfinite, B)
    @test B[1, 1, 1] == 0
end
