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
