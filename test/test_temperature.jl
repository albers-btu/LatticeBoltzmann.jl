using Test
using LatticeBoltzmann
using KernelAbstractions
using Unitful

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
            host[n] = TYPE_S | TYPE_T
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

@testset "plain TYPE_S does not impose Tm" begin
    @test TEMPERATURE && SURFACE
    Nx, Ny, Nz = 12, 12, 16
    Hfill = 10
    Tcold, Tm = 0.2f0, 1.0f0
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, Λ=0, T_avg=Tm,
                  backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Tcold, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    LatticeBoltzmann.initialize!(model)
    run!(model, 80)
    LatticeBoltzmann.moments!(model)
    T = Array(model.domains[1].T.data)
    fl = Array(model.domains[1].flags.data)
    nbot = lbm_n(6, 6, 2, Nx, Ny)
    nedge = lbm_n(2, 2, 2, Nx, Ny)
    Tif = Float32[T[n] for n in eachindex(fl) if (fl[n] & TYPE_SU) == TYPE_I]
    @info "plain TYPE_S" Tbot=T[nbot] Tedge=T[nedge] Tif_min=minimum(Tif) Tif_max=maximum(Tif)
    @test T[nbot] ≈ Tcold atol=0.05f0
    @test T[nedge] ≈ Tcold atol=0.08f0
    @test minimum(Tif) > 0
    @test maximum(Tif) < Tm
end

@testset "TYPE_S g bounce-back cools the pad; gas is adiabatic" begin
    @test TEMPERATURE && SURFACE
    Nx, Ny, Nz = 12, 8, 16
    Hfill = 10
    Tcold, Thot = 0.2f0, 1.0f0
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, Λ=0, T_avg=Thot,
                  backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Thot, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S | TYPE_T
            Th[n] = Tcold
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    LatticeBoltzmann.initialize!(model)
    run!(model, 80)
    LatticeBoltzmann.moments!(model)
    d = model.domains[1]
    TA = Array(d.T.data)
    fl = Array(d.flags.data)
    nbot = lbm_n(6, 4, 2, Nx, Ny)
    nmid = lbm_n(6, 4, 6, Nx, Ny)
    Tif = Float32[TA[n] for n in eachindex(fl) if (fl[n] & TYPE_SU) == TYPE_I]
    @info "TYPE_S BB" Tbot=TA[nbot] Tmid=TA[nmid] Tif_min=minimum(Tif) Tif_max=maximum(Tif)
    @test TA[nbot] < Thot - 0.15f0
    @test TA[nbot] > Tcold - 0.05f0
    @test minimum(Tif) > 0
    @test maximum(Tif) < Thot + 0.25f0
    @test all(isfinite, TA)
end

@testset "TYPE_S|TYPE_H Robin plate cools the pad" begin
    @test TEMPERATURE && SURFACE
    Nx, Ny, Nz = 12, 8, 16
    Hfill = 10
    Tcold, Thot = 0.2f0, 1.0f0
    hconv = 8.0f0
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, Λ=0, T_avg=Thot,
                  backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Thot, Nx * Ny * Nz)
    hh = zeros(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1
            host[n] = TYPE_S | TYPE_H
            Th[n] = Tcold
            hh[n] = hconv
        elseif z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].h.data, hh)
    LatticeBoltzmann.initialize!(model)
    run!(model, 80)
    LatticeBoltzmann.moments!(model)
    TA = Array(model.domains[1].T.data)
    nbot = lbm_n(6, 4, 2, Nx, Ny)
    @info "Robin plate" Tbot=TA[nbot]
    @test TA[nbot] < Thot - 0.15f0
    @test TA[nbot] > Tcold - 0.05f0
    @test all(isfinite, TA)
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
    @test LatticeBoltzmann.cell_enthalpy(Tm - 0.1f0, 1.0f0, Λ) ≈ Tm - 0.1f0
    @test LatticeBoltzmann.cell_enthalpy(Tm, 0.0f0, Λ) ≈ Tm + Λ
    @test LatticeBoltzmann.cell_enthalpy(Tm, 0.6f0, Λ) ≈ Tm + Λ * 0.4f0
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
    γ = 0.4f0
    T0 = 1.2f0
    H = LatticeBoltzmann.cell_enthalpy(T0, 0.0f0, Λ, γ)
    T, fl = LatticeBoltzmann.invert_enthalpy(H, Tm, Tm, Λ, γ)
    @test fl == 1
    @test T ≈ T0 atol=1.0f-5
    Hs = LatticeBoltzmann.sensible_H(Tm, γ)
    T, fl = LatticeBoltzmann.invert_enthalpy(Hs + 0.3f0, Tm, Tm, Λ, γ)
    @test T ≈ Tm
    @test fl ≈ 0.3f0 / Λ
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
    @test LatticeBoltzmann.omega_T_from_alpha(1.0f-6) <= 1.999f0
    @test LatticeBoltzmann.omega_from_nu(1.0f-8) <= 1.999f0
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
            host[n] = TYPE_S | TYPE_T
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

# Periodic TYPE_F box: wrap_coord, no walls, no gas. Closed for energy.
function _periodic_T_box(Nx, Ny, Nz; ν=0.1f0, α=0.2f0, Λ=0.0f0, Qv=0.0f0,
                         T0=1.0f0, amp=0.0f0, Tm=1.0f0, γ_s=0.0f0, γ_l=0.0f0)
    model = Model(Nx, Ny, Nz, ν; α=α, fz=0, σ=0, Λ=Λ, Ts=Tm, Tl=Tm,
                  T_avg=Tm, γ_s=γ_s, γ_l=γ_l, backend=CPU(), workgroup=64)
    N = Nx * Ny * Nz
    host = fill(TYPE_F, N)
    Th = fill(T0, N)
    fsh = ones(Float32, N)
    Qh = fill(Qv, N)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        Th[n] = T0 + amp * sin(2.0f0 * Float32(π) * Float32(x) / Float32(Nx))
        fsh[n] = LatticeBoltzmann.fs_from_T(Th[n], Tm, Tm)
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].fs.data, fsh)
    copyto!(model.domains[1].Q.data, Qh)
    LatticeBoltzmann.initialize!(model)
    return model
end

@testset "closed energy: periodic box conserves H" begin
    @test TEMPERATURE
    Nx, Ny, Nz = 12, 8, 8
    model = _periodic_T_box(Nx, Ny, Nz; amp=0.2f0)
    E0 = enthalpy(model)
    run!(model, 80)
    E1 = enthalpy(model)
    @info "closed energy Q=0" E0 E1
    @test isapprox(E1, E0; rtol=2.0f-5, atol=2.0f-4)
    m = mass_budget(model)
    @info "closed mass Q=0" m.M m.M0 m.residual
    @test isapprox(m.M, m.M0; rtol=2.0f-5, atol=2.0f-4)
    @test isapprox(m.residual, 0; atol=5.0f-4)
    @test m.evap == 0 && m.powder == 0
end

@testset "closed energy: ΔH equals ΣQ" begin
    @test TEMPERATURE
    Nx, Ny, Nz = 12, 8, 8
    Qv = 5.0f-4
    nsteps = 40
    model = _periodic_T_box(Nx, Ny, Nz; Qv=Qv, amp=0.1f0)
    E0 = enthalpy(model)
    Qin = heat_source(model)
    @test Qin ≈ Qv * Nx * Ny * Nz
    run!(model, nsteps)
    E1 = enthalpy(model)
    @info "closed energy Q" E0 E1 ΔH=(E1-E0) nQ=(nsteps * Qin)
    @test isapprox(E1 - E0, nsteps * Qin; rtol=2.0f-4, atol=2.0f-3)
    b = energy_budget(model)
    @test isapprox(b.Q, nsteps * Qin; rtol=2.0f-4, atol=2.0f-3)
    @test isapprox(b.residual, 0; atol=5.0f-3, rtol=1.0f-4)
    @test b.rad == 0 && b.evap == 0 && b.wall == 0
end

@testset "closed energy: latent heating still closes" begin
    @test TEMPERATURE
    Nx, Ny, Nz = 12, 8, 8
    Tm, Λ, Qv = 1.0f0, 0.4f0, 2.0f-3
    nsteps = 50
    model = _periodic_T_box(Nx, Ny, Nz; Λ=Λ, Qv=Qv, T0=0.95f0, amp=0.0f0, Tm=Tm)
    E0 = enthalpy(model)
    Qin = heat_source(model)
    run!(model, nsteps)
    E1 = enthalpy(model)
    d = model.domains[1]
    fsA = Array(d.fs.data)
    @info "closed energy latent" E0 E1 ΔH=(E1-E0) nQ=(nsteps * Qin) fsmin=minimum(fsA)
    @test isapprox(E1 - E0, nsteps * Qin; rtol=5.0f-4, atol=5.0f-3)
    @test E1 > E0
    b = energy_budget(model)
    @test isapprox(b.residual, 0; atol=5.0f-2, rtol=1.0f-3)
end

@testset "closed energy: cp(T) Q is ΔH not ΔT" begin
    @test TEMPERATURE
    Nx, Ny, Nz = 12, 8, 8
    γ, T0, Qv = 0.4f0, 1.5f0, 5.0f-4
    nsteps = 40
    model = _periodic_T_box(Nx, Ny, Nz; Qv=Qv, T0=T0, amp=0.0f0, γ_s=γ, γ_l=γ)
    Qin = heat_source(model)
    run!(model, nsteps)
    b = energy_budget(model)
    TA = Array(model.domains[1].T.data)
    ΔT = sum(TA) / length(TA) - T0
    @info "cp(T) energy" b.H b.H0 b.Q b.residual ΔT nQ=(nsteps * Qin)
    @test isapprox(b.residual, 0; atol=5.0f-3, rtol=1.0f-4)
    @test isapprox(b.H - b.H0, nsteps * Qin; rtol=2.0f-4, atol=2.0f-3)
    @test ΔT > 0
    @test ΔT < nsteps * Qv * 0.95f0
end

@testset "open energy: radiation and cold plate" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 12, 8, 16
    Hfill = 10
    Tinf, Thot = 1.0f0, 1.8f0
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, Λ=0.0f0,
                  T_avg=Tinf, C_rad=0.02f0, T_rad=Tinf,
                  backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Thot, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    LatticeBoltzmann.initialize!(model)
    run!(model, 25)
    b = energy_budget(model)
    @info "open energy rad" b.H b.H0 b.rad b.wall b.residual ΔH=(b.H-b.H0)
    @test b.rad > 0
    @test b.H < b.H0
    @test abs(b.residual) < abs(b.H - b.H0)

    Tplate, Tpad = 0.2f0, 1.0f0
    model2 = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, Λ=0.0f0,
                   T_avg=Tpad, backend=CPU(), workgroup=64)
    host2 = zeros(UInt8, Nx * Ny * Nz)
    Th2 = fill(Tpad, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1
            host2[n] = TYPE_S | TYPE_T
            Th2[n] = Tplate
        elseif z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host2[n] = TYPE_S
            Th2[n] = Tplate
        elseif z <= Hfill
            host2[n] = TYPE_F
        end
    end
    copyto!(model2.domains[1].flags.data, host2)
    copyto!(model2.domains[1].T.data, Th2)
    LatticeBoltzmann.initialize!(model2)
    run!(model2, 40)
    w = energy_budget(model2)
    @info "open energy wall" w.H w.H0 w.wall w.residual ΔH=(w.H-w.H0)
    @test w.wall > 0
    @test w.H < w.H0
    @test abs(w.residual) < abs(w.H - w.H0)
end

@testset "open mass: msrc matches ΔM and energy powder" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 12, 8, 8
    Tm, Tp = 1.0f0, 0.4f0
    S0 = 5.0f-4
    nsteps = 40
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, Λ=0.0f0,
                  T_avg=Tm, T_p=Tp, backend=CPU(), workgroup=64)
    N = Nx * Ny * Nz
    copyto!(model.domains[1].flags.data, fill(TYPE_F, N))
    copyto!(model.domains[1].T.data, fill(Tm, N))
    copyto!(model.domains[1].msrc.data, fill(S0, N))
    LatticeBoltzmann.initialize!(model)
    run!(model, nsteps)
    m = mass_budget(model)
    e = energy_budget(model)
    @info "open mass msrc" m.M m.M0 m.powder m.evap m.residual e.powder
    @test m.powder > 0
    @test m.evap == 0
    @test isapprox(m.M - m.M0, m.powder; atol=2.0f-3, rtol=2.0f-4)
    @test isapprox(m.residual, 0; atol=2.0f-3)
    @test isapprox(e.powder, Tp * m.powder; atol=2.0f-3, rtol=2.0f-4)
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
    Qe_cold = LatticeBoltzmann.evaporative_dT(1.0f0, 8.0f0, 1.5f0, 0.01f0, 0.5f0, 30.0f0)
    ps_cold = LatticeBoltzmann.p_sat(1.0f0, 1.5f0, 0.5f0, 30.0f0)
    @test Qe_cold > 0
    @test Qe_cold < 1.0f-4
    @test Qe_cold ≈ 0.01f0 * ps_cold / sqrt(1.0f0) * 8.0f0
    @test LatticeBoltzmann.evaporative_dT(1.6f0, 0.0f0, 1.5f0, 0.01f0, 0.5f0, 30.0f0) == 0
    Qe = LatticeBoltzmann.evaporative_dT(1.6f0, 8.0f0, 1.5f0, 0.01f0, 0.5f0, 30.0f0)
    @test Qe > 0
    # Superheat is 0.1. The boiling flux is larger, so the step crosses T_v
    # instead of stopping on it. It still cannot remove more than superheat
    # plus that boiling flux.
    Qboil = 0.01f0 * 0.5f0 / sqrt(1.5f0) * 8.0f0
    @test Qe > 1.6f0 - 1.5f0
    @test Qe ≤ (1.6f0 - 1.5f0) + Qboil + 1.0f-5
    Qhot = LatticeBoltzmann.evaporative_dT(4.0f0, 8.0f0, 1.5f0, 0.01f0, 0.5f0, 30.0f0)
    @test Qhot ≤ (4.0f0 - 1.5f0) + Qboil + 1.0f-4
    @test Qhot > 4.0f0 - 1.5f0
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

@testset "small liquid source builds a face-connected column" begin
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
    # 0.008 per step at the center. Half a cell takes well over a hundred
    # steps, so this run has to accumulate; one step cannot open the face.
    run!(model, 250)
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
    # g = 0 and σ = 0, so the gaussian peak stays where it was deposited.
    # A column is face-connected when metal above the plate has metal on
    # the cell directly under it. A corner link leaves a gas gap.
    n_gap = 0
    for y in 2:(Ny - 1), x in 2:(Nx - 1)
        under = false
        for z in 2:(Nz - 1)
            n = lbm_n(x, y, z, Nx, Ny)
            su = fl[n] & TYPE_SU
            ismet = su == TYPE_I || su == TYPE_F || su == TYPE_IF || su == TYPE_GI
            if ismet && z > Hfill + 1 && !under
                n_gap += 1
            end
            under = ismet
        end
    end
    @info "mass source column" M0 M1 hC hE n_gap
    @test M1 > M0 + 1
    @test hC > Float32(Hfill) + 1.2f0
    @test hC > hE + 0.4f0
    @test hC < Float32(Nz - 1)
    @test n_gap == 0
    mb = mass_budget(model)
    @test mb.powder > 0
    @test mb.M > mb.M0
end

@testset "liquid surplus opens only the leading cell" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 16, 8, 20
    Hfill = 12
    Tm = 1.0f0
    model = Model(Nx, Ny, Nz, 0.1f0; α = 0.0f0, β = 0.0f0, fz = 0.0f0, σ = 0.0f0,
                  Λ = 0.0f0, Ts = Tm, Tl = Tm, T_avg = Tm, backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Tm, Nx * Ny * Nz)
    fsh = zeros(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    d = model.domains[1]
    copyto!(d.flags.data, host)
    copyto!(d.T.data, Th)
    copyto!(d.fs.data, fsh)
    LatticeBoltzmann.initialize!(model)
    xc, yc = Nx ÷ 2, Ny ÷ 2
    fl = Array(d.flags.data)
    zI = 0
    for z in 1:Nz
        n = lbm_n(xc, yc, z, Nx, Ny)
        (fl[n] & TYPE_SU) == TYPE_I && (zI = z)
    end
    @test zI == Hfill + 1
    nI = lbm_n(xc, yc, zI, Nx, Ny)
    ρn = Array(d.ρ.data)[nI]
    mh = Array(d.mass.data)
    ϕh = Array(d.ϕ.data)
    # A filled interface (φ = 1) carrying a full extra cell. One step opens
    # only +z; the diagonals stay gas.
    mh[nI] = 2 * ρn
    ϕh[nI] = 1
    copyto!(d.mass.data, mh)
    copyto!(d.ϕ.data, ϕh)
    copyto!(d.fs.data, zeros(Float32, length(mh)))
    copyto!(d.massex.data, zeros(Float32, length(mh)))
    Md = sum(Array(d.mass.data))
    run!(model, 2)
    fl = Array(d.flags.data)
    ϕ = Array(d.ϕ.data)
    nup = lbm_n(xc, yc, zI + 1, Nx, Ny)
    ndiag = lbm_n(xc + 1, yc, zI + 1, Nx, Ny)
    @test (fl[nI] & TYPE_SU) == TYPE_I
    @test ϕ[nI] > 0.7f0
    @test (fl[nup] & TYPE_SU) == TYPE_I
    @test ϕ[nup] > 0.5f0
    @test (fl[ndiag] & TYPE_SU) == TYPE_G
    @test sum(Array(d.mass.data)) ≈ Md rtol=1.0f-3
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
    mb = mass_budget(model)
    @test mb.powder > 0
    @test isapprox(mb.M, mb.M0 + mb.powder - mb.evap; atol=0.05f0)
    @test abs(mb.residual) < 0.05f0
end

@testset "powder absorbs the beam; molten and liquid landings join mass" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 16, 12, 24
    Hfill = 10
    Tm = 1.0f0
    Tcold = 0.3f0
    si_H = 0.001
    Lpad = Hfill - 2
    m = si_H / Lpad
    st = 0.1 * m^2 / 7.5e-6
    units = Units(Lpad, 0.05, 1, si_H, 0.05 * m / st, 8000.0; T=Float32, K=1000.0f0, cp=500.0f0)
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, Λ=0.2f0, Ts=Tm, Tl=Tm,
                  T_avg=Tm, T_v=80.0f0, τ_p=1.0f5, T_p=Tcold, backend=CPU(), workgroup=64)
    model.units = units
    d = model.domains[1]
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Tcold, Nx * Ny * Nz)
    fsh = ones(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(d.flags.data, host)
    copyto!(d.T.data, Th)
    copyto!(d.fs.data, fsh)
    LatticeBoltzmann.initialize!(model)
    xc = Float32(Nx) / 2
    yc = Float32(Ny) / 2
    model.laser = Laser(units; P=40.0, w=3.0, x=xc, y=yc, z=Float32(Nz) - 1.1f0,
                        n_re=3.27, n_im=4.48, nrays=9, every=1, skin=1)
    d_m = 75.0e-6
    m_one = 8000.0 * π * d_m^3 / 6
    model.powder_jet = PowderJet(units; mdot=0.0, w=2.0, v=4.0,
                                 d=d_m / units.m, x=xc, y=yc, z=Float32(Nz) - 1.2f0,
                                 nparcels=1, nmax=4, enabled=false)
    J = model.powder_jet
    J.alive[1] = true
    J.px[1] = xc
    J.py[1] = yc
    J.pz[1] = Float32(Nz) - 4
    J.pm[1] = Float32(m_one / units.kg)
    J.pT[1] = Tcold
    J.alive[2] = true
    J.px[2] = xc
    J.py[2] = yc
    J.pz[2] = Float32(Nz) - 0.2f0
    J.pm[2] = J.pm[1]
    J.pT[2] = Tcold
    LatticeBoltzmann.heat_powder_beam!(model, d)
    d1 = J.pT[1] - Tcold
    @test d1 > 0
    @test J.pT[2] == Tcold
    @test 0 < J.absorbed < model.laser.P
    J.pT[1] = Tcold
    model.laser.every = 4
    LatticeBoltzmann.heat_powder_beam!(model, d)
    d4 = J.pT[1] - Tcold
    @test 3.5f0 < d4 / d1 < 4.5f0
    shadow = J.absorbed
    J.pT[1] = Tm
    d.T_v = Tm
    LatticeBoltzmann.heat_powder_beam!(model, d)
    @test J.pT[1] == Tm
    # At boiling the particle takes no more heat, but its cross section still shades.
    @test J.absorbed > 0
    @test J.absorbed ≈ shadow rtol=1.0f-5
    d.T_v = 80.0f0
    J.pT[1] = Tcold
    model.laser.every = 1
    LatticeBoltzmann.heat_powder_beam!(model, d)
    absorbed = J.absorbed
    LatticeBoltzmann.deposit_laser!(model, d)
    Qs = sum(Array(d.Q.data))
    J.absorbed = 0
    LatticeBoltzmann.deposit_laser!(model, d)
    Qf = sum(Array(d.Q.data))
    @test Qf > 0
    @test Qs < Qf
    @test Qs ≈ Qf * (1 - absorbed / model.laser.P) rtol=1.0f-4

    fl = Array(d.flags.data)
    Th = Array(d.T.data)
    fsh = Array(d.fs.data)
    zI = 0
    for z in 1:Nz
        n = lbm_n(round(Int, xc), round(Int, yc), z, Nx, Ny)
        (fl[n] & TYPE_SU) == TYPE_I && (zI = z)
    end
    @test zI > Hfill
    for n in eachindex(fl)
        su = fl[n] & TYPE_SU
        if su == TYPE_I || su == TYPE_F
            Th[n] = Tm + 0.2f0
            fsh[n] = 0.0f0
        end
    end
    copyto!(d.T.data, Th)
    copyto!(d.fs.data, fsh)
    fill!(d.Q.data, 0.02f0)
    J.alive .= false
    J.alive[1] = true
    J.px[1] = xc
    J.py[1] = yc
    J.pz[1] = Float32(zI) + 1.6f0
    J.pvx[1] = 0
    J.pvy[1] = 0
    J.pvz[1] = -4
    J.v = 4
    J.pm[1] = 0.35f0
    J.pT[1] = Tcold
    J.enabled = false
    M0 = sum(Array(d.mass.data))
    Mp0 = sum(Array(d.mp.data))
    Q0 = sum(Array(d.Q.data))
    LatticeBoltzmann.advance_powder_jet!(model, d, false)
    M1 = sum(Array(d.mass.data))
    Mp1 = sum(Array(d.mp.data))
    Q1 = sum(Array(d.Q.data))
    @test M1 > M0 + 0.2f0
    @test abs(Mp1 - Mp0) < 1.0f-6
    @test Q1 < Q0
    J.alive .= false
    LatticeBoltzmann.advance_powder_jet!(model, d, false)
    Q2 = sum(Array(d.Q.data))
    @test Q2 ≈ Q0 atol=1.0f-5

    for n in eachindex(fl)
        su = fl[n] & TYPE_SU
        if su == TYPE_I || su == TYPE_F
            Th[n] = Tcold
            fsh[n] = 1.0f0
        end
    end
    copyto!(d.T.data, Th)
    copyto!(d.fs.data, fsh)
    J.alive[1] = true
    J.px[1] = xc
    J.py[1] = yc
    J.pz[1] = Float32(zI) + 1.6f0
    J.pT[1] = Tm + 0.4f0
    Mh = sum(Array(d.mass.data))
    Mph = sum(Array(d.mp.data))
    LatticeBoltzmann.advance_powder_jet!(model, d, true)
    @test sum(Array(d.mass.data)) > Mh + 0.2f0
    @test abs(sum(Array(d.mp.data)) - Mph) < 1.0f-6
    J.alive[1] = true
    J.px[1] = xc
    J.py[1] = yc
    J.pz[1] = Float32(zI) + 1.6f0
    J.pT[1] = Tcold
    Mc = sum(Array(d.mass.data))
    Mpc = sum(Array(d.mp.data))
    LatticeBoltzmann.advance_powder_jet!(model, d, true)
    @test abs(sum(Array(d.mass.data)) - Mc) < 1.0f-5
    @test sum(Array(d.mp.data)) > Mpc + 0.2f0
end

@testset "hot solid holds powder; frozen overflow builds the cell above" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 16, 12, 20
    Hfill = 8
    Tm = 1.0f0
    Tcold = 0.2f0
    Twarm = 0.95f0
    si_H = 0.001
    Lpad = Hfill - 2
    m = si_H / Lpad
    st = 0.1 * m^2 / 7.5e-6
    units = Units(Lpad, 0.05, 1, si_H, 0.05 * m / st, 8000.0; T=Float32, K=1000.0f0, cp=500.0f0)
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.0f0, fz=0, σ=0, Λ=0.2f0, Ts=Tm, Tl=Tm,
                  T_avg=Tm, τ_p=2.0f0, T_p=Tcold, n_hydro=1, backend=CPU(), workgroup=64)
    model.units = units
    d = model.domains[1]
    @test d.T_stick ≈ 0.9f0 * Tm
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Twarm, Nx * Ny * Nz)
    fsh = ones(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
            Th[n] = Tcold
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(d.flags.data, host)
    copyto!(d.T.data, Th)
    copyto!(d.fs.data, fsh)
    LatticeBoltzmann.initialize!(model)
    xc = 4
    yc = 6
    nw = lbm_n(xc, yc, Hfill - 1, Nx, Ny)
    nc = lbm_n(xc + 8, yc, Hfill - 1, Nx, Ny)
    Th = Array(d.T.data)
    fsh = Array(d.fs.data)
    mph = Array(d.mp.data)
    Th[nc] = Tcold
    fsh[nw] = 1
    fsh[nc] = 1
    mph[nw] = 0.5f0
    mph[nc] = 0.5f0
    copyto!(d.T.data, Th)
    copyto!(d.fs.data, fsh)
    copyto!(d.mp.data, mph)
    fill!(d.Q.data, 0)
    # One step: the cold cell's populations are still the warm init, so the
    # next collide restores T and a longer run is no longer a decay test.
    run!(model, 1)
    mph = Array(d.mp.data)
    @test mph[nw] ≈ 0.5f0 atol=1.0f-5
    @test mph[nc] ≈ 0.5f0 * exp(-0.5f0) rtol=1.0f-4

    model = Model(Nx, Ny, Nz, 0.1f0; α=0.05f0, fz=0, σ=0, Λ=0.2f0, Ts=Tm, Tl=Tm,
                  T_avg=Tm, τ_p=1.0f5, T_p=Tcold, n_hydro=1, backend=CPU(), workgroup=64)
    model.units = units
    d = model.domains[1]
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Tcold, Nx * Ny * Nz)
    fsh = ones(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(d.flags.data, host)
    copyto!(d.T.data, Th)
    copyto!(d.fs.data, fsh)
    LatticeBoltzmann.initialize!(model)
    xc = round(Int, Nx / 2)
    yc = round(Int, Ny / 2)
    fl = Array(d.flags.data)
    zI = 0
    for z in 1:Nz
        n = lbm_n(xc, yc, z, Nx, Ny)
        (fl[n] & TYPE_SU) == TYPE_I && (zI = z)
    end
    @test zI == Hfill + 1
    nI = lbm_n(xc, yc, zI, Nx, Ny)
    @test Array(d.fs.data)[nI] > 0.999f0
    model.powder_jet = PowderJet(units; mdot=0.0, w=2.0, v=4.0,
                                 x=Float32(xc), y=Float32(yc), z=Float32(zI) + 2,
                                 nparcels=1, nmax=2, enabled=false)
    J = model.powder_jet
    J.alive[1] = true
    J.px[1] = Float32(xc)
    J.py[1] = Float32(yc)
    J.pz[1] = Float32(zI) + 1.6f0
    J.pvx[1] = 0
    J.pvy[1] = 0
    J.pvz[1] = -4
    J.v = 4
    J.pm[1] = 1.2f0
    J.pT[1] = Tm + 0.2f0
    M0 = sum(Array(d.mass.data))
    LatticeBoltzmann.advance_powder_jet!(model, d, true)
    @test sum(Array(d.mass.data)) > M0 + 1.0f0
    Md = sum(Array(d.mass.data))
    # The interface density settles over a couple of steps. The frozen cell
    # keeps the surplus until it is half a cell, then the new cell pulls it.
    run!(model, 4)
    fl = Array(d.flags.data)
    ϕ = Array(d.ϕ.data)
    fsh = Array(d.fs.data)
    nup = lbm_n(xc, yc, zI + 1, Nx, Ny)
    ndiag = lbm_n(xc + 1, yc, zI + 1, Nx, Ny)
    @test (fl[nI] & TYPE_SU) == TYPE_I
    @test (fl[nup] & TYPE_SU) == TYPE_I
    @test ϕ[nI] > 0.9f0
    @test ϕ[nup] > 0.3f0
    @test fsh[nup] < 0.5f0
    @test (fl[ndiag] & TYPE_SU) == TYPE_G
    @test sum(Array(d.mass.data)) ≈ Md rtol=1.0f-3
end

@testset "boiling parcel mixes into the cell instead of dumping the enthalpy gap" begin
    Ts = 1.0f0
    Λ = 0.25f0
    γs = 0.0f0
    γl = 0.0f0
    Tc = 0.7f0
    pT = 1.85f0
    pmass = 0.56f0
    ρn = 1.0f0
    m0 = 1.0f0
    Q0 = 0.01f0
    mp = Float32[0]
    mass = Float32[m0]
    Q = Float32[Q0]
    qhold = Float32[0]
    Tfield = Float32[Tc]
    fs = Float32[0]
    ρ = Float32[ρn]
    flags = UInt8[TYPE_F]
    h_base = LatticeBoltzmann.cell_enthalpy(Tc, 0.0f0, Λ, γl)
    h_p = LatticeBoltzmann.sensible_H(pT, γl) + Λ
    h_mix = (m0 * h_base + pmass * h_p) / (m0 + pmass)
    undiluted = (pmass / ρn) * (h_p - h_base)
    dm, dE = LatticeBoltzmann._deposit_parcel!(
        mp, mass, Q, qhold, Tfield, fs, ρ, flags, 1,
        pmass, pT, 1.0f0, Ts, Λ, γs, γl)
    @test dm ≈ pmass
    @test dE ≈ pmass * h_p
    @test mass[1] ≈ m0 + pmass
    @test mp[1] == 0
    @test Q[1] ≈ Q0 + (h_mix - h_base)
    @test qhold[1] ≈ h_base - h_mix
    @test abs(Q[1] - Q0) < abs(undiluted) - 0.15f0
    m1 = m0 + pmass
    h_mix2 = (m1 * h_mix + pmass * h_p) / (m1 + pmass)
    LatticeBoltzmann._deposit_parcel!(
        mp, mass, Q, qhold, Tfield, fs, ρ, flags, 1,
        pmass, pT, 1.0f0, Ts, Λ, γs, γl)
    @test mass[1] ≈ m1 + pmass
    @test Q[1] ≈ Q0 + (h_mix2 - h_base) atol=1.0f-5
    @test qhold[1] ≈ h_base - h_mix2 atol=1.0f-5
    @test abs(Q[1] - Q0) < 2 * abs(undiluted) - 0.4f0

    # Solid receiver: latent heat is Λ(1−fs), so a full Λ is not already in the cell.
    Tc_s = 0.2f0
    pT_s = 1.2f0
    pm_s = 1.2f0
    m_s = 0.5f0
    mass[1] = m_s
    Q[1] = 0
    qhold[1] = 0
    Tfield[1] = Tc_s
    fs[1] = 1
    h_base_s = LatticeBoltzmann.cell_enthalpy(Tc_s, 1.0f0, Λ, γs)
    h_p_s = LatticeBoltzmann.sensible_H(pT_s, γl) + Λ
    h_mix_s = (m_s * h_base_s + pm_s * h_p_s) / (m_s + pm_s)
    LatticeBoltzmann._deposit_parcel!(
        mp, mass, Q, qhold, Tfield, fs, ρ, flags, 1,
        pm_s, pT_s, 1.0f0, Ts, Λ, γs, γl)
    @test mass[1] ≈ m_s + pm_s
    @test Q[1] ≈ h_mix_s - h_base_s atol=1.0f-5
    @test h_mix_s > Ts
    @test h_mix_s < Ts + Λ
end

@testset "cold powder join stays above the feed temperature" begin
    # A solid cell at Ts that has held a full cell of cold powder. The old
    # debit (T − T_p + Λ) drove H below 0, about −550 K for IN625.
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 16, 12, 16
    Hfill = 8
    Tm = 1.0f0
    Tcold = 0.185f0
    γs = 0.31165f0
    Λ = 0.24755f0
    model = Model(Nx, Ny, Nz, 0.02f0; α=0.0f0, fz=0, σ=0, Λ=Λ,
                  Ts=Tm, Tl=Tm, T_avg=Tm, γ_s=γs, γ_l=0.0f0,
                  τ_p=1.0f5, T_p=Tcold, n_hydro=1, backend=CPU(), workgroup=64)
    d = model.domains[1]
    N = Nx * Ny * Nz
    host = zeros(UInt8, N)
    Th = fill(Tcold, N)
    fsh = ones(Float32, N)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(d.flags.data, host)
    copyto!(d.T.data, Th)
    copyto!(d.fs.data, fsh)
    initialize!(model)
    xc, yc = Nx ÷ 2, Ny ÷ 2
    fl = Array(d.flags.data)
    zI = 0
    for z in 1:Nz
        n = lbm_n(xc, yc, z, Nx, Ny)
        (fl[n] & TYPE_SU) == TYPE_I && (zI = z)
    end
    @test zI == Hfill + 1
    nI = lbm_n(xc, yc, zI, Nx, Ny)
    ρn = Array(d.ρ.data)[nI]
    Th = Array(d.T.data)
    fsh = Array(d.fs.data)
    mh = Array(d.mass.data)
    mph = Array(d.mp.data)
    Qh = Array(d.Q.data)
    Th[nI] = Tm
    fsh[nI] = 1
    mh[nI] = ρn
    mph[nI] = ρn
    Qh[nI] = 0.05f0
    copyto!(d.T.data, Th)
    copyto!(d.fs.data, fsh)
    copyto!(d.mass.data, mh)
    copyto!(d.mp.data, mph)
    copyto!(d.Q.data, Qh)
    n0 = nI - 1
    x = n0 % Nx
    y = (n0 ÷ Nx) % Ny
    z = n0 ÷ (Nx * Ny)
    LatticeBoltzmann.store_geq!(d.gi.data, nI, x, y, z, Tm, 0.0f0, 0.0f0, 0.0f0,
                                N, Nx, Ny, Nz, false, Float32)
    M0 = sum(Array(d.mass.data)) + sum(Array(d.mp.data))
    LatticeBoltzmann.step!(model)
    Ti = Array(d.T.data)[nI]
    h_base = LatticeBoltzmann.cell_enthalpy(Tm, 1.0f0, Λ, γs)
    h_now = h_base + 0.05f0
    h_p = LatticeBoltzmann.sensible_H(Tcold, γs)
    h_mix = (ρn * h_now + ρn * h_p) / (ρn + ρn)
    Tmix = LatticeBoltzmann.invert_enthalpy(h_mix, Tm, Tm, Λ, γs)[1]
    # The extra cell is published as massex for the neighbor's next pull.
    M1 = sum(Array(d.mass.data)) + sum(Array(d.mp.data)) + sum(Array(d.massex.data))
    @test Ti > Tcold
    @test Ti < Tm
    @test Ti ≈ Tmix atol=0.05f0
    @test Array(d.mp.data)[nI] == 0
    @test M1 ≈ M0 rtol=1.0f-4
end

function cold_plate_model(Nx, Ny, Nz, Hfill)
    Tm = 1.0f0
    Tcold = 0.2f0
    si_H = 0.001
    Lpad = Hfill - 2
    m = si_H / Lpad
    st = 0.1 * m^2 / 7.5e-6
    units = Units(Lpad, 0.05, 1, si_H, 0.05 * m / st, 8000.0; T=Float32, K=1000.0f0, cp=500.0f0)
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.0f0, fz=0, σ=0, Λ=0.2f0, Ts=Tm, Tl=Tm,
                  T_avg=Tm, τ_p=1.0f5, T_p=Tcold, n_hydro=1, backend=CPU(), workgroup=64)
    model.units = units
    d = model.domains[1]
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(Tcold, Nx * Ny * Nz)
    fsh = ones(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(d.flags.data, host)
    copyto!(d.T.data, Th)
    copyto!(d.fs.data, fsh)
    LatticeBoltzmann.initialize!(model)
    return model, Tm, Tcold
end

@testset "small surplus stays on a flat frozen interface" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 16, 12, 20
    Hfill = 8
    model, _, _ = cold_plate_model(Nx, Ny, Nz, Hfill)
    d = model.domains[1]
    xc = round(Int, Nx / 2)
    yc = round(Int, Ny / 2)
    fl = Array(d.flags.data)
    zI = 0
    for z in 1:Nz
        n = lbm_n(xc, yc, z, Nx, Ny)
        (fl[n] & TYPE_SU) == TYPE_I && (zI = z)
    end
    @test zI == Hfill + 1
    nI = lbm_n(xc, yc, zI, Nx, Ny)
    ρn = Array(d.ρ.data)[nI]
    mh = Array(d.mass.data)
    mh[nI] = 1.2f0 * ρn
    copyto!(d.mass.data, mh)
    fsh = ones(Float32, length(mh))
    copyto!(d.fs.data, fsh)
    Md = sum(Array(d.mass.data))
    run!(model, 2)
    fl = Array(d.flags.data)
    mh = Array(d.mass.data)
    nup = lbm_n(xc, yc, zI + 1, Nx, Ny)
    ndiag = lbm_n(xc + 1, yc, zI + 1, Nx, Ny)
    @test (fl[nI] & TYPE_SU) == TYPE_I
    @test (fl[nup] & TYPE_SU) == TYPE_G
    @test (fl[ndiag] & TYPE_SU) == TYPE_G
    @test mh[nI] > ρn
    @test mh[nI] ≈ 1.2f0 * ρn rtol=1.0f-3
    @test sum(mh) ≈ Md rtol=1.0f-3
end

@testset "tilted normal opens the two cube faces" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 16, 12, 20
    Hfill = 8
    model, _, _ = cold_plate_model(Nx, Ny, Nz, Hfill)
    d = model.domains[1]
    xc = round(Int, Nx / 2)
    yc = round(Int, Ny / 2)
    fl = Array(d.flags.data)
    zI = 0
    for z in 1:Nz
        n = lbm_n(xc, yc, z, Nx, Ny)
        (fl[n] & TYPE_SU) == TYPE_I && (zI = z)
    end
    @test zI == Hfill + 1
    nI = lbm_n(xc, yc, zI, Nx, Ny)
    nm = lbm_n(xc - 1, yc, zI, Nx, Ny)
    # Metal only behind the interface, so the outward normal points up and +x.
    for y in 2:Ny-1, x in 2:Nx-1
        n = lbm_n(x, y, zI, Nx, Ny)
        (n == nI || n == nm) && continue
        fl[n] = TYPE_G
    end
    copyto!(d.flags.data, fl)
    ρn = Array(d.ρ.data)[nI]
    mh = Array(d.mass.data)
    ϕh = Array(d.ϕ.data)
    for y in 2:Ny-1, x in 2:Nx-1
        n = lbm_n(x, y, zI, Nx, Ny)
        if fl[n] == TYPE_G
            mh[n] = 0
            ϕh[n] = 0
        end
    end
    mh[nI] = 2 * ρn
    ϕh[nI] = 1
    ϕh[nm] = 1
    copyto!(d.mass.data, mh)
    copyto!(d.ϕ.data, ϕh)
    copyto!(d.massex.data, zeros(Float32, length(mh)))
    copyto!(d.fs.data, ones(Float32, length(mh)))
    Md = sum(Array(d.mass.data))
    run!(model, 2)
    fl = Array(d.flags.data)
    ϕ = Array(d.ϕ.data)
    nup = lbm_n(xc, yc, zI + 1, Nx, Ny)
    ndiag = lbm_n(xc + 1, yc, zI + 1, Nx, Ny)
    nside = lbm_n(xc + 1, yc, zI, Nx, Ny)
    nyp = lbm_n(xc, yc + 1, zI, Nx, Ny)
    nym = lbm_n(xc, yc - 1, zI, Nx, Ny)
    # n = (1, 0, 1)/√2. The cube faces +z and +x each have aperture 1/√2
    # and tie, so both advance. The diagonal is not a face: its link carries
    # no volume flux, and that cell stays gas. ±y sit on the plate, so ∂y = 0
    # is a flat direction with a back side, not a free pair.
    @test (fl[nI] & TYPE_SU) == TYPE_I
    @test (fl[nup] & TYPE_SU) == TYPE_I
    @test (fl[nside] & TYPE_SU) == TYPE_I
    @test ϕ[nup] > 0.2f0
    @test ϕ[nside] > 0.2f0
    @test (fl[ndiag] & TYPE_SU) == TYPE_G
    @test (fl[nyp] & TYPE_SU) == TYPE_G
    @test (fl[nym] & TYPE_SU) == TYPE_G
    @test ϕ[nI] > 0.9f0
    @test sum(Array(d.mass.data)) ≈ Md rtol=1.0f-3
end

@testset "lip with gas above and below fills downward" begin
    # A one-cell overhang. Centered ∂zφ is zero. Neither face is backed by
    # bulk, so each aperture is φ − φ_nb and the cell under the lip opens
    # with the forward face.
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 16, 12, 24
    Hfill = 8
    model, _, _ = cold_plate_model(Nx, Ny, Nz, Hfill)
    d = model.domains[1]
    xc = round(Int, Nx / 2)
    yc = round(Int, Ny / 2)
    fl = Array(d.flags.data)
    zI = 0
    for z in 1:Nz
        n = lbm_n(xc, yc, z, Nx, Ny)
        (fl[n] & TYPE_SU) == TYPE_I && (zI = z)
    end
    @test zI == Hfill + 1
    zL = zI + 2
    nL = lbm_n(xc, yc, zL, Nx, Ny)
    nBack = lbm_n(xc - 1, yc, zL, Nx, Ny)
    nDown = lbm_n(xc, yc, zL - 1, Nx, Ny)
    nUp = lbm_n(xc, yc, zL + 1, Nx, Ny)
    nFwd = lbm_n(xc + 1, yc, zL, Nx, Ny)
    ρn = Array(d.ρ.data)[lbm_n(xc, yc, zI, Nx, Ny)]
    mh = Array(d.mass.data)
    ϕh = Array(d.ϕ.data)
    ρh = Array(d.ρ.data)
    fl[nL] = TYPE_I
    fl[nBack] = TYPE_I
    mh[nL] = 2 * ρn
    mh[nBack] = ρn
    ϕh[nL] = 1
    ϕh[nBack] = 1
    ρh[nL] = ρn
    ρh[nBack] = ρn
    copyto!(d.flags.data, fl)
    copyto!(d.mass.data, mh)
    copyto!(d.ϕ.data, ϕh)
    copyto!(d.ρ.data, ρh)
    copyto!(d.massex.data, zeros(Float32, length(mh)))
    fsh = zeros(Float32, length(mh))
    copyto!(d.fs.data, fsh)
    N = Nx * Ny * Nz
    for ncell in (nL, nBack)
        n0 = ncell - 1
        x = n0 % Nx
        y = (n0 ÷ Nx) % Ny
        z = n0 ÷ (Nx * Ny)
        LatticeBoltzmann.store_feq!(d.fi.data, ncell, x, y, z, ρn, 0.0f0, 0.0f0, 0.0f0,
                                    model.weights, model.velocities, N, Nx, Ny, Nz, Val(false))
        LatticeBoltzmann.store_feq!(d.fi.data, ncell, x, y, z, ρn, 0.0f0, 0.0f0, 0.0f0,
                                    model.weights, model.velocities, N, Nx, Ny, Nz, Val(true), Val(true))
    end
    Md = sum(Array(d.mass.data))
    run!(model, 2)
    fl = Array(d.flags.data)
    ϕ = Array(d.ϕ.data)
    @test (fl[nL] & TYPE_SU) == TYPE_I
    @test (fl[nDown] & TYPE_SU) == TYPE_I
    @test (fl[nUp] & TYPE_SU) == TYPE_I
    @test (fl[nFwd] & TYPE_SU) == TYPE_I
    @test ϕ[nDown] > 0.1f0
    @test sum(Array(d.mass.data)) ≈ Md rtol=1.0f-3
end

@testset "unsupported sheet fills the gap down to the plate" begin
    # φ = 1 with the same fill above and below has ∇φ = 0. Neither side is
    # bulk, so each face is open by φ − φ_nb and the excess walks down.
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 16, 12, 28
    Hfill = 8
    model, Tm, _ = cold_plate_model(Nx, Ny, Nz, Hfill)
    d = model.domains[1]
    xc = round(Int, Nx / 2)
    yc = round(Int, Ny / 2)
    fl = Array(d.flags.data)
    zI = 0
    for z in 1:Nz
        n = lbm_n(xc, yc, z, Nx, Ny)
        (fl[n] & TYPE_SU) == TYPE_I && (zI = z)
    end
    @test zI == Hfill + 1
    zS = zI + 5
    ρn = Array(d.ρ.data)[lbm_n(xc, yc, zI, Nx, Ny)]
    mh = Array(d.mass.data)
    ϕh = Array(d.ϕ.data)
    ρh = Array(d.ρ.data)
    Th = Array(d.T.data)
    fsh = Array(d.fs.data)
    # Wide enough that the center's horizontal neighbors are not the rim.
    # The rim has gas beyond it; an interior cell only has gas above and below.
    for dy in -2:2, dx in -2:2
        n = lbm_n(xc + dx, yc + dy, zS, Nx, Ny)
        fl[n] = TYPE_I
        mh[n] = 20 * ρn
        ϕh[n] = 1
        ρh[n] = ρn
        Th[n] = Tm
        fsh[n] = 1
    end
    copyto!(d.flags.data, fl)
    copyto!(d.mass.data, mh)
    copyto!(d.ϕ.data, ϕh)
    copyto!(d.ρ.data, ρh)
    copyto!(d.T.data, Th)
    copyto!(d.fs.data, fsh)
    copyto!(d.massex.data, zeros(Float32, length(mh)))
    N = Nx * Ny * Nz
    for dy in -2:2, dx in -2:2
        ncell = lbm_n(xc + dx, yc + dy, zS, Nx, Ny)
        n0 = ncell - 1
        x = n0 % Nx
        y = (n0 ÷ Nx) % Ny
        z = n0 ÷ (Nx * Ny)
        LatticeBoltzmann.store_feq!(d.fi.data, ncell, x, y, z, ρn, 0.0f0, 0.0f0, 0.0f0,
                                    model.weights, model.velocities, N, Nx, Ny, Nz, Val(false))
        LatticeBoltzmann.store_feq!(d.fi.data, ncell, x, y, z, ρn, 0.0f0, 0.0f0, 0.0f0,
                                    model.weights, model.velocities, N, Nx, Ny, Nz, Val(true), Val(true))
    end
    run!(model, 40)
    fl = Array(d.flags.data)
    mh = Array(d.mass.data)
    # The free surface walks down. Symmetry can leave the exact center
    # under half a cell, so the layer is counted across the patch.
    for z in (zI + 1):zS
        cnt = 0
        for dy in -3:3, dx in -3:3
            su = fl[lbm_n(xc + dx, yc + dy, z, Nx, Ny)] & TYPE_SU
            if su == TYPE_I || su == TYPE_F
                cnt += 1
            end
        end
        @test cnt > 0
    end
    nmid = lbm_n(xc, yc, zI + 2, Nx, Ny)
    @test (fl[nmid] & TYPE_SU) == TYPE_I || (fl[nmid] & TYPE_SU) == TYPE_F
    @test mh[nmid] > 0.2f0 * ρn
end

@testset "powder jet draws a d10/d50/d90 distribution" begin
    @test SURFACE && TEMPERATURE
    Nx, Ny, Nz = 16, 12, 24
    Hfill = 10
    Tm = 1.0f0
    Tcold = 0.3f0
    si_H = 0.001
    Lpad = Hfill - 2
    m = si_H / Lpad
    st = 0.1 * m^2 / 7.5e-6
    units = Units(Lpad, 0.05, 1, si_H, 0.05 * m / st, 8000.0; T=Float32, K=1000.0f0, cp=500.0f0)
    d10 = 45.0e-6 / units.m
    d50 = 75.0e-6 / units.m
    d90 = 125.0e-6 / units.m
    dmin = 20.0e-6 / units.m
    dmax = 150.0e-6 / units.m
    J = PowderJet(units; mdot=1.0e-4, w=2.0, v=1.0, d=d50,
                  d10=d10, d50=d50, d90=d90, dmin=dmin, dmax=dmax,
                  x=8.0, y=6.0, z=20.0, nparcels=64, nmax=64, enabled=true)
    LatticeBoltzmann._spawn_parcels!(J, 0.01f0, 0.3f0)
    ds = [J.pd[i] for i in 1:J.nmax if J.alive[i]]
    @test length(ds) == 64
    @test all(dmin .<= ds .<= dmax)
    @test maximum(ds) > minimum(ds) * 1.2

    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, Λ=0.2f0, Ts=Tm, Tl=Tm,
                  T_avg=Tm, T_v=80.0f0, τ_p=1.0f5, T_p=Tcold, backend=CPU(), workgroup=64)
    model.units = units
    d = model.domains[1]
    host = zeros(UInt8, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(d.flags.data, host)
    LatticeBoltzmann.initialize!(model)
    xc = Float32(Nx) / 2
    yc = Float32(Ny) / 2
    model.laser = Laser(units; P=40.0, w=3.0, x=xc, y=yc, z=Float32(Nz) - 1.1f0,
                        n_re=3.27, n_im=4.48, nrays=9, every=1, skin=1)
    model.powder_jet = PowderJet(units; mdot=0.0, w=2.0, v=4.0, d=0.0,
                                 x=xc, y=yc, z=Float32(Nz) - 1.2f0,
                                 nparcels=1, nmax=2, enabled=false)
    Js = model.powder_jet
    Js.alive[1] = true
    Js.px[1] = xc
    Js.py[1] = yc
    Js.pz[1] = Float32(Nz) - 4
    Js.pm[1] = 1.0f-3
    Js.pT[1] = Tcold
    d_small = Float32(50.0e-6 / units.m)
    d_large = Float32(100.0e-6 / units.m)
    Js.pd[1] = d_small
    LatticeBoltzmann.heat_powder_beam!(model, d)
    a_small = Js.absorbed
    Js.pd[1] = 0
    Js.d = d_small
    Js.pT[1] = Tcold
    LatticeBoltzmann.heat_powder_beam!(model, d)
    @test Js.absorbed ≈ a_small rtol=1.0f-5
    Js.d = 0
    Js.pd[1] = d_large
    Js.pT[1] = Tcold
    LatticeBoltzmann.heat_powder_beam!(model, d)
    a_large = Js.absorbed
    @test a_small > 0
    @test a_large > 0
    @test a_small / a_large ≈ d_large / d_small rtol=1.0f-4
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

@testset "ray that steps into bulk uses the Fresnel skin" begin
    # A filled surface lets the ray miss the cut and enter TYPE_F.
    # The whole ray in one cell is nskin/A too hot and is what blew up Tim_DED.
    A0 = LatticeBoltzmann.fresnel_absorptance(1.0f0, 3.27f0, 4.48f0)
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
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z==1 || z==Nz || x==1 || x==Nx || y==1 || y==Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    LatticeBoltzmann.initialize!(model)
    flags = model.domains[1].flags.data
    for n in eachindex(flags)
        if (flags[n] & TYPE_SU) == TYPE_I
            flags[n] = (flags[n] & ~TYPE_SU) | TYPE_G
        end
    end
    las = Laser(units; P=200.0, w=2.0, x=Nx/2, y=Ny/2, z=Float32(Nz)-1.1f0,
                nrays=9, max_bounce=4, skin=3)
    model.laser = las
    fill!(model.domains[1].Q.data, 0)
    LatticeBoltzmann.deposit_laser!(model, model.domains[1])
    QA = Array(model.domains[1].Q.data)
    qfac = LatticeBoltzmann.laser_qfac(units)
    Pabs = sum(QA) / qfac
    @test isfinite(Pabs)
    @test 0.20 * las.P < Pabs < 0.60 * las.P
    @test abs(Pabs - A0 * las.P) / las.P < 0.15
    # One cell used to receive a whole ray. Skin 3 keeps every cell under that.
    @test maximum(QA) < las.P * qfac / 3
end

@testset "filled interface keeps the ray and the parcel" begin
    # φ = 1 puts the cut on the top face. A downward step that has already
    # crossed that face must deposit here, not in the TYPE_F cell below.
    @test TEMPERATURE && SURFACE
    Nx, Ny, Nz = 16, 12, 20
    Hfill = 8
    Tm = 1.0f0
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.0f0, fz=0, σ=0, Λ=0.2f0, Ts=Tm, Tl=Tm,
                  T_avg=Tm, backend=CPU(), workgroup=64)
    d = model.domains[1]
    host = zeros(UInt8, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(d.flags.data, host)
    LatticeBoltzmann.initialize!(model)
    xc = Nx ÷ 2
    yc = Ny ÷ 2
    flags = Array(d.flags.data)
    zI = 0
    for z in 1:Nz
        (flags[lbm_n(xc, yc, z, Nx, Ny)] & TYPE_SU) == TYPE_I && (zI = z)
    end
    @test zI == Hfill + 1
    nI = lbm_n(xc, yc, zI, Nx, Ny)
    nF = lbm_n(xc, yc, zI - 1, Nx, Ny)
    ϕ = Array(d.ϕ.data)
    phij = LatticeBoltzmann.gather_phi_d3q27(ϕ, ϕ[nI], xc - 1, yc - 1, zI - 1, Nx, Ny, Nz)
    # Gas half, ray heading further into the gas: the cut is behind the sample.
    hit_gas, _, _ = LatticeBoltzmann.plic_hit(
        ϕ[nI], phij,
        Float32(xc), Float32(yc), Float32(zI) + 0.2f0,
        0.0f0, 0.0f0, 1.0f0,
        Float32(xc), Float32(yc), Float32(zI))
    @test !hit_gas
    for n in eachindex(ϕ)
        (flags[n] & TYPE_SU) == TYPE_I && (ϕ[n] = 1)
    end
    copyto!(d.ϕ.data, ϕ)
    phij = LatticeBoltzmann.gather_phi_d3q27(ϕ, 1.0f0, xc - 1, yc - 1, zI - 1, Nx, Ny, Nz)
    hit, t, nϕ = LatticeBoltzmann.plic_hit(
        1.0f0, phij,
        Float32(xc), Float32(yc), Float32(zI) + 0.5f0 - 1.0f-5,
        0.0f0, 0.0f0, -1.0f0,
        Float32(xc), Float32(yc), Float32(zI))
    @test hit
    @test t == 0
    @test nϕ[3] > 0.9f0

    model.laser = Laser(model.units; P=200.0, w=2.0,
                         x=Float32(xc), y=Float32(yc), z=Float32(Nz) - 1.1f0,
                         nrays=1, max_bounce=1, every=1, skin=1)
    fill!(d.Q.data, 0)
    LatticeBoltzmann.deposit_laser!(model, d)
    QA = Array(d.Q.data)
    @test QA[nI] > 0
    @test QA[nF] == 0

    mass = Array(d.mass.data)
    mI = mass[nI]
    mF = mass[nF]
    mp = zeros(Float32, length(mass))
    qhold = zeros(Float32, length(mass))
    _, _, _, _, dm, _ = LatticeBoltzmann._walk_parcel!(
        mp, mass, zeros(Float32, length(mass)), qhold,
        Array(d.T.data), Array(d.fs.data), Array(d.ρ.data), flags, ϕ, 1.0f0,
        Float32(xc), Float32(yc), Float32(zI) + 1.2f0,
        0.0f0, 0.0f0, -1.0f0,
        5.0f0, 0.4f0, Tm + 0.2f0, Tm, 0.2f0, 0.0f0, 0.0f0,
        Nx, Ny, Nz)
    @test dm ≈ 0.4f0
    @test mass[nI] ≈ mI + 0.4f0
    @test mass[nF] ≈ mF

    # Gas on both sides cancels Parker–Youngs. A full cell is still metal.
    sheet = ntuple(i -> i == 1 ? 1.0f0 : 0.0f0, 27)
    hit_sheet, t_sheet, n_sheet = LatticeBoltzmann.plic_hit(
        1.0f0, sheet,
        1.0f0, 1.0f0, 1.0f0,
        0.0f0, 0.0f0, -1.0f0,
        1.0f0, 1.0f0, 1.0f0)
    @test hit_sheet
    @test t_sheet == 0
    @test n_sheet[3] > 0.9f0
    miss_sheet, _, _ = LatticeBoltzmann.plic_hit(
        0.5f0, sheet,
        1.0f0, 1.0f0, 1.0f0,
        0.0f0, 0.0f0, -1.0f0,
        1.0f0, 1.0f0, 1.0f0)
    @test !miss_sheet
end

@testset "wild interface population stays finite" begin
    # One ± pair at ±30 is the anti-bounce state that used to overflow Tim_DED.
    # Opposite signs keep ρ ~ 1, so the equilibrium replacement stays under the cap.
    @test LatticeBoltzmann.population_ok(0.4f0)
    @test !LatticeBoltzmann.population_ok(30.0f0)
    @test !LatticeBoltzmann.population_ok(NaN32)
    @test !LatticeBoltzmann.population_ok(Inf32)
    @test LatticeBoltzmann.bound_population(30.0f0, 0.11f0) == 0.11f0
    @test LatticeBoltzmann.bound_population(-0.2f0, 0.11f0) == -0.2f0
    @test TEMPERATURE && SURFACE
    Nx, Ny, Nz = 16, 12, 20
    Hfill = 8
    Tm = 1.0f0
    model = Model(Nx, Ny, Nz, 0.01f0; α=0.0f0, fz=0, σ=0, Λ=0.2f0, Ts=Tm, Tl=Tm,
                  T_avg=Tm, backend=CPU(), workgroup=64)
    d = model.domains[1]
    host = zeros(UInt8, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        elseif z <= Hfill
            host[n] = TYPE_F
        end
    end
    copyto!(d.flags.data, host)
    LatticeBoltzmann.initialize!(model)
    xc = Nx ÷ 2
    yc = Ny ÷ 2
    flags = Array(d.flags.data)
    zI = 0
    for z in 1:Nz
        (flags[lbm_n(xc, yc, z, Nx, Ny)] & TYPE_SU) == TYPE_I && (zI = z)
    end
    @test zI == Hfill + 1
    @test Int(d.t) == 0
    nI = lbm_n(xc, yc, zI, Nx, Ny)
    nG = lbm_n(xc, yc, zI + 1, Nx, Ny)
    # Even load of +z: f₊ at the interface, f₋ in the gas cell above.
    N = Nx * Ny * Nz
    i = 6
    ip = LatticeBoltzmann.f_index(nI, i + 1, N)
    im = LatticeBoltzmann.f_index(nG, i, N)
    fi = d.fi.data
    pair = fi[ip] + fi[im]
    fi[ip] = 30
    fi[im] = pair - 30
    for _ in 1:4
        LatticeBoltzmann.step!(model)
    end
    fa = Array(d.fi.data)
    @test all(isfinite, fa)
    @test maximum(abs, fa) <= 2.5f0
    @test all(isfinite, Array(d.T.data))
    @test all(isfinite, Array(d.u.data))
    @test all(isfinite, Array(d.gi.data))
    @test all(isfinite, Array(d.mass.data))
end

@testset "gas face does not multiply T by normal speed" begin
    # The gas never collides, so EsotericPull used to feed this cell its own
    # outgoing population. That population is geq(T, +u_n); the return link
    # has to be geq(T, −u_n). The difference is T·u_n, and with |u| on c_s
    # the interface temperature ran away in a few dozen steps.
    @test TEMPERATURE && SURFACE
    function iface_after(uz::Float32, n_hydro::Int, steps::Int)
        Nx, Ny, Nz = 16, 12, 20
        Hfill = 8
        T0 = 1.2f0
        model = Model(Nx, Ny, Nz, 0.02f0;
            α = 0.22f0, fz = 0, σ = 0, Λ = 0, Ts = 0.2f0, Tl = 0.2f0,
            T_avg = 1.0f0, γ_s = 0, γ_l = 0, K0 = 0, Λ_v = 0,
            n_hydro = n_hydro, backend = CPU(), workgroup = 64)
        d = model.domains[1]
        N = Nx * Ny * Nz
        host = zeros(UInt8, N)
        Th = fill(T0, N)
        for z in 1:Nz, y in 1:Ny, x in 1:Nx
            n = lbm_n(x, y, z, Nx, Ny)
            if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
                host[n] = TYPE_S
            elseif z <= Hfill
                host[n] = TYPE_F
            end
        end
        copyto!(d.flags.data, host)
        copyto!(d.T.data, Th)
        initialize!(model)
        for n in 1:N
            fl = d.flags.data[n]
            (fl & TYPE_S) != 0 && continue
            (fl & TYPE_SU) == TYPE_G && continue
            n0 = n - 1
            x = n0 % Nx
            y = (n0 ÷ Nx) % Ny
            z = n0 ÷ (Nx * Ny)
            ρn = max(d.ρ.data[n], 1.0f0)
            LatticeBoltzmann.store_feq!(d.fi.data, n, x, y, z, ρn, 0.0f0, 0.0f0, uz,
                model.weights, model.velocities, N, Nx, Ny, Nz, Val(false), Val(true))
            LatticeBoltzmann.store_geq!(d.gi.data, n, x, y, z, T0, 0.0f0, 0.0f0, uz,
                N, Nx, Ny, Nz, Val(false), Float32, Val(true))
            d.u.data[n, 3] = uz
            d.T.data[n] = T0
            d.fs.data[n] = 0
        end
        xc, yc = Nx ÷ 2, Ny ÷ 2
        Ti = T0
        umax = 0.0f0
        for _ in 1:steps
            LatticeBoltzmann.step!(model)
            zI = 0
            for z in 1:Nz
                n = lbm_n(xc, yc, z, Nx, Ny)
                (d.flags.data[n] & TYPE_SU) == TYPE_I && (zI = z)
            end
            zI == 0 && return Ti, umax
            nI = lbm_n(xc, yc, zI, Nx, Ny)
            Ti = d.T.data[nI]
            umax = max(umax, abs(d.u.data[nI, 3]))
        end
        return Ti, umax
    end
    # The imposed 0.45 is above the liquid-interface cap. Collide rewrites the
    # Guo force so the streamed speed is at most 0.2; the stored moment is the
    # half-force velocity, about 0.31 here. The same flux opens the cell above,
    # so the sample moves. Temperature of that face stays at the initial value.
    T1, u1 = iface_after(0.45f0, 1, 6)
    @test u1 > 0.25f0
    @test u1 < 0.5f0
    @test isapprox(T1, 1.2f0; atol=0.05f0)
    # Even hydro count: gi parity is the outer step, not the hydro substep.
    T2, u2 = iface_after(0.45f0, 2, 6)
    @test u2 > 0.2f0
    @test u2 < 0.5f0
    @test isapprox(T2, 1.2f0; atol=0.08f0)
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

@testset "beam and powder export as 1/e² cylinders" begin
    using WriteVTK
    Nx, Ny, Nz = 8, 8, 16
    Hfill = 4
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, Λ=0, T_avg=1.0f0,
                  backend=CPU(), workgroup=64)
    flags = fill(TYPE_G, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            flags[n] = TYPE_S
        elseif z <= Hfill
            flags[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, flags)
    U = model.units
    model.laser = Laser(U; P=10.0, w=1.5, x=4, y=4, z=Nz - 1.2,
                         nrays=3, max_bounce=1, every=1, skin=1)
    model.powder_jet = PowderJet(U; mdot=1.0e-4, w=2.0, v=1.0, d=0.4,
                                  x=6.0, y=4.0, z=Nz - 1.5,
                                  dir=(-0.4, 0.0, -1.0),
                                  nparcels=2, enabled=true)
    mktempdir() do d
        export!(model; dir=d)
        @test isfile(joinpath(d, "rays.pvd"))
        @test isfile(joinpath(d, "beam.pvd"))
        @test isfile(joinpath(d, "powder.pvd"))
        ray = read(joinpath(d, "rays_00000000.vtp"), String)
        @test occursin("Lines", ray)
        @test !occursin("Strips", ray)
        @test !occursin("Polys", ray)
        beam = read(joinpath(d, "beam_00000000.vtp"), String)
        @test occursin("Strips", beam)
        @test occursin("Polys", beam)
        @test occursin("NumberOfPoints=\"48\"", beam)
        @test !occursin("Lines", beam)
        pow = read(joinpath(d, "powder_00000000.vtp"), String)
        @test occursin("Strips", pow)
        @test occursin("Polys", pow)
        @test occursin("mdot", pow)
    end
    xs = Float32[]; ys = Float32[]; zs = Float32[]
    strips = MeshCell{PolyData.Strips, Vector{Int}}[]
    polys = MeshCell{PolyData.Polys, Vector{Int}}[]
    dx = Float32(U.m)
    @test LatticeBoltzmann._vtk_add_cylinder!(
        xs, ys, zs, strips, polys, 4.0, 4.0, Nz - 1.2, 0.0, 0.0, -1.0, 1.5,
        flags, Nx, Ny, Nz, dx)
    zring = zs[25:48]
    @test minimum(zring) > Float32(Hfill - 1) * dx
    @test maximum(zring) < Float32(Hfill + 2) * dx
    r = hypot(xs[1] - (4 - 1) * dx, ys[1] - (4 - 1) * dx)
    @test isapprox(r, 1.5f0 * dx; rtol=1.0f-5)
end

@testset "powder feed splits across a ring of jets" begin
    pkg = dirname(dirname(pathof(LatticeBoltzmann)))
    include(joinpath(pkg, "input", "DED_powder.jl"))
    @test si_powder_tilt == 0u"°"
    @test powder_jet_azimuths == (0u"°",)
    include(joinpath(pkg, "input", "Tim_DED_powder.jl"))
    @test si_powder_tilt == 30u"°"
    @test powder_jet_azimuths == (90u"°", 210u"°", 330u"°")
    degs = sort([mod(ustrip(u"°", a), 360.0) for a in powder_jet_azimuths])
    gaps = (degs[2] - degs[1], degs[3] - degs[2], degs[1] + 360 - degs[3])
    @test all(isapprox.(gaps, 120.0; atol=1e-6))
    for a in powder_jet_azimuths
        deg = mod(ustrip(u"°", a), 360.0)
        @test min(abs(deg), abs(deg - 180), abs(deg - 360)) > 1
    end

    Nx, Ny, Nz = 32, 32, 24
    Hfill = 8
    model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, fz=0, σ=0, Λ=0.2f0,
                  Ts=1.0f0, Tl=1.0f0, T_avg=1.0f0, T_v=80.0f0,
                  τ_p=1.0f5, T_p=0.3f0, n_hydro=1, backend=CPU(), workgroup=64)
    flags = fill(TYPE_G, Nx * Ny * Nz)
    Th = fill(0.3f0, Nx * Ny * Nz)
    fsh = ones(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            flags[n] = TYPE_S
        elseif z <= Hfill
            flags[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, flags)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].fs.data, fsh)
    initialize!(model)
    U = model.units
    mdot = 9.5u"g/minute"
    want = Float32(ustrip(u"kg/s", mdot))
    jets = make_powder_jets(U; mdot=mdot, n=length(powder_jet_azimuths),
                             w=2.0, v=8.0, d=0.4, nparcels=4, nmax=32, enabled=false)
    @test length(jets) == 3
    @test all(J -> J.nparcels == 4, jets)
    @test all(J -> isapprox(J.mdot, want / 3; rtol=1.0f-5), jets)
    @test isapprox(sum(J.mdot for J in jets), want; rtol=1.0f-5)
    one = make_powder_jets(U; mdot=mdot, n=1, w=2.0, v=8.0, d=0.4, nparcels=4)
    @test isapprox(one[1].mdot, want; rtol=1.0f-5)
    @test_throws ArgumentError make_powder_jets(U; mdot=mdot, n=0, w=1.0, v=1.0)

    x_las, y_las = 16.0, 16.0
    z_noz, z_aim = Float32(Nz) - 1.5f0, Float32(Hfill)
    tilt = ustrip(u"rad", si_powder_tilt)
    az = ntuple(i -> ustrip(u"rad", powder_jet_azimuths[i]), length(powder_jet_azimuths))
    place_powder_jets!(jets, x_las, y_las, z_noz, z_aim, Nx, Ny, 1, tilt, az)
    drop = float(z_noz) - float(z_aim)
    R = drop * tan(tilt)
    angs = Float64[]
    for (J, ψ) in zip(jets, az)
        @test isapprox(J.x, x_las + R * cos(ψ); atol=1e-3)
        @test isapprox(J.y, y_las + R * sin(ψ); atol=1e-3)
        @test isapprox(J.z, z_noz; atol=1e-4)
        @test 2.5 <= J.x <= Nx - 1.5
        @test 2.5 <= J.y <= Ny - 1.5
        @test abs(J.y - y_las) > 0.4 * R
        tox = x_las - J.x
        toy = y_las - J.y
        toz = z_aim - J.z
        nrm = hypot(tox, toy, toz)
        @test isapprox(J.dx * tox + J.dy * toy + J.dz * toz, nrm; atol=1e-4)
        @test isapprox(atan(hypot(J.x - x_las, J.y - y_las), J.z - z_aim), tilt; atol=1e-4)
        push!(angs, atan(J.y - y_las, J.x - x_las))
    end
    for i in 1:3
        dψ = abs(mod(angs[i] - angs[mod1(i + 1, 3)] + π, 2π) - π)
        @test isapprox(dψ, 2π / 3; atol=1e-4)
    end

    d = model.domains[1]
    model.laser = Laser(U; P=10.0, w=2.0, x=x_las, y=y_las, z=z_noz,
                         nrays=3, max_bounce=1, every=1, skin=1)
    for J in jets
        J.alive[1] = true
        J.px[1] = x_las
        J.py[1] = y_las
        J.pz[1] = (z_noz + z_aim) / 2
        J.pm[1] = 1.0f0
        J.pT[1] = 0.3f0
        J.pd[1] = 0.2f0
    end
    probe = LatticeBoltzmann._powder_raw_shadow(jets[1], model.laser, U)
    @test probe !== nothing && sum(probe) > 0
    share_m = Float32(jets[1].pm[1] * (0.1f0 * model.laser.P) / sum(probe))
    for J in jets
        J.pm[1] = share_m
    end
    model.powder_jet = jets[1]
    LatticeBoltzmann.heat_powder_beam!(model, d)
    alone = jets[1].absorbed
    @test isapprox(alone, 0.1f0 * model.laser.P; rtol=1.0f-3)
    model.powder_jet = jets
    LatticeBoltzmann.heat_powder_beam!(model, d)
    @test isapprox(jets[1].absorbed, alone; rtol=1.0f-4)
    @test isapprox(jets[2].absorbed, alone; rtol=1.0f-4)
    @test isapprox(jets[3].absorbed, alone; rtol=1.0f-4)
    @test jets[1].absorbed + jets[2].absorbed + jets[3].absorbed < model.laser.P
    for J in jets
        J.pm[1] = share_m * 1.0f4
        J.pT[1] = 0.3f0
    end
    LatticeBoltzmann.heat_powder_beam!(model, d)
    shadowed = sum(J.absorbed for J in jets)
    @test isapprox(shadowed, Float32(model.laser.P); rtol=1.0f-4)
    @test all(J -> isapprox(J.absorbed, Float32(model.laser.P) / 3; rtol=1.0f-3), jets)
    @test isapprox(LatticeBoltzmann.powder_beam_transmit(model), 0.0f0; atol=1.0f-4)
    model.powder_jet = jets[1]
    LatticeBoltzmann.heat_powder_beam!(model, d)
    @test isapprox(jets[1].absorbed, Float32(model.laser.P); rtol=1.0f-4)

    model.powder_jet = jets
    for J in jets
        fill!(J.alive, false)
        J.pm[1] = 0
    end
    set_powder_enabled!(jets, false)
    mktempdir() do dir
        export!(model; dir)
        @test !isfile(joinpath(dir, "powder.pvd"))
    end
    set_powder_enabled!(jets, true)
    mktempdir() do dir
        export!(model; dir)
        pow = read(joinpath(dir, "powder_00000000.vtp"), String)
        @test occursin("Strips", pow)
        @test occursin("Polys", pow)
        @test occursin("mdot", pow)
        @test occursin("NumberOfPoints=\"144\"", pow)
    end
    model.powder_jet = one[1]
    one[1].enabled = true
    set_powder_jet_position!(one[1], x_las, y_las, z_noz)
    aim_powder_jet!(one[1], x_las, y_las, z_aim)
    mktempdir() do dir
        export!(model; dir)
        pow = read(joinpath(dir, "powder_00000000.vtp"), String)
        @test occursin("NumberOfPoints=\"48\"", pow)
    end

    model.powder_jet = jets
    set_powder_enabled!(jets, true)
    place_powder_jets!(jets, x_las, y_las, z_noz, z_aim, Nx, Ny, 1, tilt, az)
    for J in jets
        fill!(J.alive, false)
    end
    run!(model, 6)
    @test mass_budget(model).powder > 0
    @test all(isfinite, Array(d.T.data))
    @test all(isfinite, Array(d.u.data))
end

# Heat collides once per outer step. An even hydro count used to reload one gi slot.
function _conduction_T(n_hydro::Int, nsteps::Int)
    Nx, Ny, Nz = 8, 8, 12
    model = Model(Nx, Ny, Nz, 0.05; α=0.05f0, β=0.0f0, fx=0.0f0, fy=0.0f0, fz=0.0f0,
                  n_hydro=n_hydro, backend=CPU(), workgroup=64)
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = ones(Float32, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz
            host[n] = TYPE_S | TYPE_T
            Th[n] = z == 1 ? 1.5f0 : 0.5f0
        else
            host[n] = TYPE_F
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    initialize!(model)
    run!(model, nsteps)
    moments!(model)
    return Array(model.domains[1].T.data)
end

@testset "temperature streams for even and odd n_hydro" begin
    @test TEMPERATURE && SURFACE
    T1 = _conduction_T(1, 24)
    T2 = _conduction_T(2, 24)
    T3 = _conduction_T(3, 24)
    nnear = lbm_n(4, 4, 2, 8, 8)
    @test T1[nnear] > 1.02f0
    @test maximum(abs.(T1 .- T2)) < 1.0f-4
    @test maximum(abs.(T1 .- T3)) < 1.0f-4
    @test all(isfinite, T2)
end

# Hertz–Knudsen is once per outer step. Recoil pressure is divided by n_hydro²
# because it is a hydro force; the evaporative coefficient has to undo that.
function _evap_interface_T(n_hydro::Int, steps::Int)
    Nx, Ny, Nz = 12, 8, 14
    Hfill = 8
    T0 = 1.30f0
    model = Model(Nx, Ny, Nz, 0.05f0; α = 0.01f0, fz = 0.0f0, β = 0.0f0, σ = 0.0f0,
                  Λ = 0.0f0, Ts = 1.0f0, Tl = 1.0f0, K0 = 0.0f0, T_avg = 1.0f0,
                  γ_s = 0.0f0, γ_l = 0.0f0,
                  # C_hk·p0 is the evaporative flux. p0 itself is the recoil pressure,
                  # so keep p0 small and put the flux in C_hk. Otherwise the gas
                  # density jump moves this thin layer and the interface index is lost.
                  Λ_v = 4.0f0, T_v = 1.60f0, C_hk = 2.5f0, p0v = 0.02f0, β_v = 12.0f0,
                  n_hydro = n_hydro, backend = CPU(), workgroup = 64)
    host = zeros(UInt8, Nx * Ny * Nz)
    Th = fill(T0, length(host))
    fsh = zeros(Float32, length(host))
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = lbm_n(x, y, z, Nx, Ny)
        if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
            Th[n] = T0
        elseif z <= Hfill
            host[n] = TYPE_F
            fsh[n] = 0
        end
    end
    copyto!(model.domains[1].flags.data, host)
    copyto!(model.domains[1].T.data, Th)
    copyto!(model.domains[1].fs.data, fsh)
    initialize!(model)
    run!(model, steps)
    TA = Array(model.domains[1].T.data)
    flags = Array(model.domains[1].flags.data)
    xc, yc = Nx ÷ 2, Ny ÷ 2
    zI = 0
    for z in 1:Nz
        (flags[lbm_n(xc, yc, z, Nx, Ny)] & TYPE_SU) == TYPE_I && (zI = z)
    end
    zI == 0 && return NaN32
    return TA[lbm_n(xc, yc, zI, Nx, Ny)]
end

@testset "evaporation cooling does not shrink with n_hydro" begin
    @test TEMPERATURE && SURFACE
    # One outer step. Later steps advect the interface, and that advection
    # depends on n_hydro, so a long run does not measure the flux coefficient.
    T1 = _evap_interface_T(1, 1)
    T9 = _evap_interface_T(9, 1)
    @info "evaporation vs n_hydro" T1 T9
    @test T1 < 1.29f0
    @test T1 > 1.20f0
    @test abs(T1 - T9) < 1.0f-3
    @test isfinite(T1) && isfinite(T9)
end
